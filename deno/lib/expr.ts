/**
 * Expressions of x as generator specs, `{expr: "a*exp(-(x-mu)^2)", params: {a: 2, mu: 1}}`
 * (lib/generators.ts): parsed here and compiled to an f64 closure of x and the params, sampled on
 * the CPU; constant subexpressions are folded. Errors name the token and its position: they go
 * back to the model that wrote the expression.
 */

import { bm } from "./boost.ts";

export const MAX_EXPR = 2048, MAX_PARAMS = 16, MAX_DEPTH = 128;

type Node =
  | { k: "num"; v: number }
  | { k: "x" }
  | { k: "par"; i: number }
  | { k: "neg"; a: Node }
  | { k: "bin"; op: string; a: Node; b: Node }
  | { k: "call"; fn: string; args: Node[] };

type F = (x: number, p: number[]) => number;

/** name: [arity, f64]. The special functions are Boost.Math's (lib/boost.ts); their poles and
 * domain errors give NaN (a gap in the curve). */
const FNS: Record<string, [number, (...a: number[]) => number]> = {
  sin: [1, Math.sin], cos: [1, Math.cos], tan: [1, Math.tan], asin: [1, Math.asin], acos: [1, Math.acos],
  atan: [1, Math.atan], atan2: [2, Math.atan2], sinh: [1, Math.sinh], cosh: [1, Math.cosh], tanh: [1, Math.tanh],
  exp: [1, Math.exp], log: [1, Math.log], log2: [1, Math.log2], log10: [1, Math.log10], sqrt: [1, Math.sqrt],
  cbrt: [1, Math.cbrt], abs: [1, Math.abs], sign: [1, Math.sign], floor: [1, Math.floor], ceil: [1, Math.ceil],
  min: [2, Math.min], max: [2, Math.max], pow: [2, Math.pow], hypot: [2, Math.hypot],
  erf: [1, bm.erf], erfc: [1, bm.erfc],
  lgamma: [1, bm.lgamma], // ln |Γ(x)|
  tgamma: [1, (x) => bm.tgamma(x)],
  digamma: [1, bm.digamma],
  zeta: [1, bm.zeta], // Riemann ζ(s)
  expint: [1, (x) => bm.expint(x)], // Ei(x)
  besselJ: [2, bm.cyl_bessel_j], // (ν, x): J_ν(x), Y_ν, I_ν, K_ν; x < 0 only for integer ν
  besselY: [2, bm.cyl_neumann],
  besselI: [2, bm.cyl_bessel_i],
  besselK: [2, bm.cyl_bessel_k],
  airyAi: [1, bm.airy_ai],
  lambertW: [1, bm.lambert_w0], // the principal branch W₀
  ellipK: [1, (k) => bm.ellint_1(k)], // complete K(k), modulus k (not m = k²)
  sinc: [1, bm.sinc_pi], // sin(x)/x
};
const CONSTS: Record<string, number> = { pi: Math.PI, e: Math.E };
const RESERVED = new Set(["x", ...Object.keys(FNS), ...Object.keys(CONSTS)]);
const CMP = new Set(["<", "<=", ">", ">=", "==", "!="]);

export interface CompiledExpr {
  names: string[]; // the params it uses, in the order of f's p
  f: F; // f64
}

/** Parses and compiles `src`, a function of x and the names in `params`. Throws an Error naming
 * the offending token and its position. */
export function compileExpr(src: string, params: string[] = []): CompiledExpr {
  if (typeof src !== "string" || !src.trim()) throw new Error("fn expr must be a non-empty string, e.g. \"exp(-x^2/2)\"");
  if (src.length > MAX_EXPR) throw new Error(`fn expr is ${src.length} characters: at most ${MAX_EXPR}`);
  if (params.length > MAX_PARAMS) throw new Error(`fn expr: ${params.length} params, at most ${MAX_PARAMS}`);
  for (const p of params) {
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(p)) throw new Error(`fn expr: param name "${p}" must be letters, digits and _, not starting with a digit`);
    if (RESERVED.has(p)) throw new Error(`fn expr: param name "${p}" is taken (x, the constants pi, e and the functions)`);
  }
  const names: string[] = [];
  let pos = 0;
  // AST depth (operators, calls, unary minus), checked as it is built: a positioned error, and
  // closures nested no deeper than that
  const depths = new WeakMap<Node, number>();
  const dep = (n: Node) => depths.get(n) ?? 0;
  const node = (n: Node, d: number): Node => {
    if (d > MAX_DEPTH) fail(`nested more than ${MAX_DEPTH} deep (each operator, call and unary minus is a level: a sum of n terms is n - 1 deep)`, tok.at);
    depths.set(n, d);
    return n;
  };
  const fail = (msg: string, at = pos): never => {
    throw new Error(`fn expr: ${msg} at character ${at + 1}: ${JSON.stringify(src.slice(Math.max(0, at - 24), at))} >>here>> ${JSON.stringify(src.slice(at, at + 24))}`);
  };
  type Tok = { t: "num" | "id" | "op" | "end"; s: string; at: number };
  const next = (): Tok => {
    while (pos < src.length && /\s/.test(src[pos])) pos += 1;
    const at = pos;
    if (pos >= src.length) return { t: "end", s: "end of input", at };
    const m = /^(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?/.exec(src.slice(pos)) ??
      /^[A-Za-z_][A-Za-z0-9_]*/.exec(src.slice(pos)) ?? /^(?:\*\*|<=|>=|==|!=|[-+*/^(),<>])/.exec(src.slice(pos));
    if (!m) fail(`unexpected character "${src[pos]}" (allowed: numbers, x, params, + - * / ^ ** < <= > >= == !=, (), function calls)`);
    pos += m![0].length;
    const s = m![0] === "**" ? "^" : m![0];
    return { t: /^[\d.]/.test(s) ? "num" : /^[A-Za-z_]/.test(s) ? "id" : "op", s, at };
  };
  let tok = next();
  const take = () => ((tok = next()), tok);
  const isOp = (s: string) => tok.t === "op" && tok.s === s;
  const expect = (s: string, what: string) => (isOp(s) ? take() : fail(`expected ${what}, got "${tok.s}"`, tok.at));

  const atom = (): Node => {
    const t = tok;
    if (t.t === "num") {
      take();
      return { k: "num", v: Number(t.s) };
    }
    if (t.t === "op" && t.s === "(") {
      take();
      const e = cmp();
      expect(")", `")" closing the "(" at character ${t.at + 1}`);
      return e;
    }
    if (t.t !== "id") return fail(t.t === "end" ? "expression ends early" : `expected a number, x, a param or a function call, got "${t.s}"`, t.at);
    take();
    if (Object.hasOwn(FNS, t.s)) {
      const [arity] = FNS[t.s];
      if (!isOp("(")) return fail(`function ${t.s} needs its arguments, ${t.s}(${arity === 1 ? "u" : "a, b"})`, tok.at);
      take();
      const args = [cmp()];
      while (isOp(",")) (take(), args.push(cmp()));
      expect(")", `")" or "," in the arguments of ${t.s}`);
      if (args.length !== arity) fail(`${t.s} takes ${arity} argument${arity > 1 ? "s" : ""}, got ${args.length}`, t.at);
      return node({ k: "call", fn: t.s, args }, Math.max(...args.map(dep)) + 1);
    }
    if (t.s === "x") return { k: "x" };
    if (Object.hasOwn(CONSTS, t.s)) return { k: "num", v: CONSTS[t.s] };
    if (params.includes(t.s)) {
      if (!names.includes(t.s)) names.push(t.s);
      return { k: "par", i: names.indexOf(t.s) };
    }
    if (isOp("(")) return fail(`unknown function "${t.s}" (known: ${Object.keys(FNS).join(" ")})`, t.at);
    return fail(`unknown name "${t.s}": not x, a param (${params.join(", ") || "none given: add params: {" + t.s + ": ...}"}), pi, e, or a function (${Object.keys(FNS).join(" ")})`, t.at);
  };
  const power = (): Node => { // right-associative, above unary minus: -x^2 = -(x^2), 2^-x
    const a = atom();
    if (isOp("^")) {
      take();
      const b = unary();
      return node({ k: "bin", op: "^", a, b }, Math.max(dep(a), dep(b)) + 1);
    }
    return a;
  };
  const unary = (): Node => {
    if (isOp("-") || isOp("+")) {
      const neg = tok.s === "-";
      take();
      const a = unary();
      return neg ? node({ k: "neg", a }, dep(a) + 1) : a;
    }
    return power();
  };
  const binary = (ops: string[], sub: () => Node) => (): Node => {
    let a = sub();
    while (tok.t === "op" && ops.includes(tok.s)) {
      const op = tok.s;
      take();
      const b = sub();
      a = node({ k: "bin", op, a, b }, Math.max(dep(a), dep(b)) + 1);
    }
    return a;
  };
  const cmp = binary([...CMP], binary(["+", "-"], binary(["*", "/"], unary)));

  const ast = cmp();
  if (tok.t !== "end") {
    fail(isOp(")") ? `unmatched ")"` : `missing operator before "${tok.s}" (write 2*x, not 2x)`, tok.at);
  }
  return { names, ...gen(ast) };
}

/** A node's closure; constant subexpressions folded. */
function gen(n: Node): { f: F; c: boolean } {
  const fold = (f: F) => {
    const v = f(0, []);
    return { f: () => v, c: true };
  };
  switch (n.k) {
    case "num": return fold(() => n.v);
    case "x": return { f: (x) => x, c: false };
    case "par": return { f: (_x, p) => p[n.i], c: false };
    case "neg": {
      const a = gen(n.a);
      const f: F = (x, p) => -a.f(x, p);
      return a.c ? fold(f) : { f, c: false };
    }
    case "call": {
      const args = n.args.map(gen);
      const [, fn] = FNS[n.fn];
      const af = args.map((a) => a.f);
      const f: F = af.length === 1 ? (x, p) => fn(af[0](x, p)) : (x, p) => fn(af[0](x, p), af[1](x, p));
      return args.every((a) => a.c) ? fold(f) : { f, c: false };
    }
    case "bin": {
      const a = gen(n.a), b = gen(n.b), af = a.f, bf = b.f;
      const f: F = {
        "+": (x: number, p: number[]) => af(x, p) + bf(x, p),
        "-": (x: number, p: number[]) => af(x, p) - bf(x, p),
        "*": (x: number, p: number[]) => af(x, p) * bf(x, p),
        "/": (x: number, p: number[]) => af(x, p) / bf(x, p),
        "^": (x: number, p: number[]) => Math.pow(af(x, p), bf(x, p)),
        "<": (x: number, p: number[]) => +(af(x, p) < bf(x, p)),
        "<=": (x: number, p: number[]) => +(af(x, p) <= bf(x, p)),
        ">": (x: number, p: number[]) => +(af(x, p) > bf(x, p)),
        ">=": (x: number, p: number[]) => +(af(x, p) >= bf(x, p)),
        "==": (x: number, p: number[]) => +(af(x, p) === bf(x, p)),
        "!=": (x: number, p: number[]) => +(af(x, p) !== bf(x, p)),
      }[n.op]!;
      return a.c && b.c ? fold(f) : { f, c: false };
    }
  }
}
