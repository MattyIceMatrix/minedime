"""Alpha genomes: formulas as expression trees, with random growth, mutation and crossover.
Blast radius: pure computation, touches no files, network, or processes."""
from .ops import OPS, WINDOWS

# A node is (name, children_tuple, window). Terminals have no children.


def to_str(n):
    name, ch, w = n
    if not ch:
        return name
    args = [to_str(c) for c in ch] + ([str(w)] if w else [])
    return f"{name}({', '.join(args)})"


def size(n):
    return 1 + sum(size(c) for c in n[1])


def depth(n):
    return 1 + max((depth(c) for c in n[1]), default=0)


def grow(rng, terminals, max_depth, p_term=0.3):
    if max_depth <= 1 or (max_depth < 4 and rng.random() < p_term):
        return (terminals[rng.integers(len(terminals))], (), None)
    name = list(OPS)[rng.integers(len(OPS))]
    _, arity, windowed = OPS[name]
    ch = tuple(grow(rng, terminals, max_depth - 1, p_term) for _ in range(arity))
    return (name, ch, int(rng.choice(WINDOWS)) if windowed else None)


def evaluate(n, data, cache):
    key = to_str(n)
    if key in cache:
        return cache[key]
    name, ch, w = n
    if not ch:
        out = data[name]
    else:
        fn = OPS[name][0]
        args = [evaluate(c, data, cache) for c in ch]
        out = fn(*args, w) if w else fn(*args)
    if len(cache) < 150:  # each entry is a full T x N panel; keep memory bounded
        cache[key] = out
    return out


def _paths(n, p=()):
    yield p
    for i, c in enumerate(n[1]):
        yield from _paths(c, p + (i,))


def _get(n, p):
    for i in p:
        n = n[1][i]
    return n


def _set(n, p, new):
    if not p:
        return new
    name, ch, w = n
    ch = list(ch)
    ch[p[0]] = _set(ch[p[0]], p[1:], new)
    return (name, tuple(ch), w)


def mutate(rng, n, terminals, max_depth):
    paths = list(_paths(n))
    p = paths[rng.integers(len(paths))]
    target = _get(n, p)
    r = rng.random()
    if r < 0.4:  # replace a subtree with a fresh one
        new = grow(rng, terminals, max(2, max_depth - len(p)))
    elif r < 0.7 and target[2]:  # nudge a lookback window
        new = (target[0], target[1], int(rng.choice(WINDOWS)))
    elif target[1]:  # swap operator for one of the same shape
        same = [k for k, v in OPS.items()
                if v[1] == len(target[1]) and v[2] == bool(target[2])]
        new = (same[rng.integers(len(same))], target[1], target[2])
    else:
        new = (terminals[rng.integers(len(terminals))], (), None)
    out = _set(n, p, new)
    return out if depth(out) <= max_depth + 1 else n


def crossover(rng, a, b, max_depth):
    pa = list(_paths(a)); pb = list(_paths(b))
    child = _set(a, pa[rng.integers(len(pa))], _get(b, pb[rng.integers(len(pb))]))
    return child if depth(child) <= max_depth + 1 else a
