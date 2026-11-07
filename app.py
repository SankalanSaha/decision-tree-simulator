from __future__ import annotations

import hashlib
import html
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Decision Tree Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styling
CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1450px;}
#MainMenu, footer {visibility: hidden;}
.hero {border-radius: 18px; padding: 1.3rem 1.8rem; margin-bottom: 1.1rem;}
.hero h1 {color:#fff; margin:0; font-size:1.85rem; font-weight:750; letter-spacing:-.02em; padding:0;}
.hero p  {color:rgba(255,255,255,.88); margin:.25rem 0 .7rem 0; font-size:.98rem;}
.pill {display:inline-block; padding:.18rem .7rem; margin:0 .4rem .2rem 0; border-radius:999px;
       background:rgba(255,255,255,.18); color:#fff; font-size:.78rem; font-weight:500;}
[data-testid="stMetric"] {background:rgba(128,128,128,.07); border:1px solid rgba(128,128,128,.22);
       padding:.75rem 1rem; border-radius:14px;}
.stTabs [data-baseweb="tab-list"] {gap:.35rem;}
.stTabs [data-baseweb="tab"] {border-radius:10px 10px 0 0; padding:.5rem 1rem; font-weight:500;}
.chip {display:inline-flex; align-items:center; gap:.4rem; margin:.15rem .7rem .15rem 0; font-size:.85rem;}
.dot {width:.8rem; height:.8rem; border-radius:50%; display:inline-block;}
.card {border:1px solid rgba(128,128,128,.25); border-radius:14px; padding:.9rem 1rem;
       background:rgba(128,128,128,.05);}
.step {border-left:3px solid #6366f1; padding:.35rem .8rem; margin:.35rem 0;
       background:rgba(99,102,241,.07); border-radius:0 10px 10px 0; font-size:.93rem;}
.step.leaf {border-color:#10b981; background:rgba(16,185,129,.09);}
.small {font-size:.82rem; opacity:.75;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

PALETTE = ["#6366F1", "#F59E0B", "#10B981", "#EF4444", "#06B6D4",
           "#A855F7", "#EC4899", "#84CC16", "#F97316", "#14B8A6"]


def wide(fn, *a, **k):
    """Call a Streamlit element full-width across Streamlit versions."""
    try:
        return fn(*a, width="stretch", **k)
    except TypeError:
        return fn(*a, use_container_width=True, **k)


def hex_mix(c: str, t: float) -> str:
    """Blend hex colour c towards white; t=1 → pure colour, t=0 → white."""
    r, g, b = (int(c[i:i + 2], 16) for i in (1, 3, 5))
    f = lambda v: int(255 - (255 - v) * t)
    return f"#{f(r):02x}{f(g):02x}{f(b):02x}"


def dist_bar(counts, names, colors, height=12) -> str:
    tot = counts.sum() or 1
    segs = "".join(
        f'<div title="{names[i]}: {int(c)}" style="width:{c / tot * 100:.2f}%;background:{colors[i]}"></div>'
        for i, c in enumerate(counts) if c > 0)
    return (f'<div style="display:flex;height:{height}px;border-radius:99px;overflow:hidden;'
            f'background:rgba(128,128,128,.2)">{segs}</div>')


def legend(names, colors) -> str:
    return "".join(f'<span class="chip"><span class="dot" style="background:{c}"></span>{n}</span>'
                   for n, c in zip(names, colors))


def style_fig(fig, h=330):
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=34, b=8),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.12))
    return fig


# Decision-tree engine (from scratch, fully vectorised per feature)
def impurity(counts: np.ndarray, n: np.ndarray, criterion: str) -> np.ndarray:
    """counts: (m, k) class counts; n: (m,) sample counts → (m,) impurity."""
    p = counts / np.maximum(np.asarray(n, float), 1)[:, None]
    if criterion == "gini":
        return 1.0 - (p ** 2).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(p > 0, p * np.log2(np.where(p > 0, p, 1.0)), 0.0).sum(1)


def scan_feature(x, Y, criterion, min_leaf=1):
    """Evaluate every candidate threshold of one feature.
    Returns (thresholds, gains) for all valid split points."""
    n = len(x)
    empty = (np.array([]), np.array([]))
    if n < 2:
        return empty
    order = np.argsort(x, kind="stable")
    xs, cum = x[order], np.cumsum(Y[order], axis=0)
    total = cum[-1]
    ln = np.arange(1, n)
    rn = n - ln
    valid = (xs[:-1] < xs[1:]) & (ln >= min_leaf) & (rn >= min_leaf)
    if not valid.any():
        return empty
    lc, ln, rn = cum[:-1][valid], ln[valid], rn[valid]
    rc = total - lc
    parent = impurity(total[None], np.array([n]), criterion)[0]
    weighted = (ln * impurity(lc, ln, criterion) + rn * impurity(rc, rn, criterion)) / n
    lo, hi = xs[:-1][valid], xs[1:][valid]
    thr = (lo + hi) / 2
    thr = np.where(thr >= hi, lo, thr)  # float-rounding guard
    return thr, parent - weighted


@dataclass
class Node:
    id: int
    depth: int
    idx: np.ndarray
    counts: np.ndarray
    impurity: float
    feature: int = -1
    threshold: float = float("nan")
    gain: float = 0.0
    left: int = -1
    right: int = -1

    @property
    def is_leaf(self): return self.left < 0
    @property
    def n(self): return len(self.idx)
    @property
    def pred(self): return int(np.argmax(self.counts))


@dataclass
class Tree:
    nodes: list
    k: int
    criterion: str
    n_features: int
    n_train: int

    @property
    def depth(self): return max(n.depth for n in self.nodes)
    @property
    def n_leaves(self): return sum(n.is_leaf for n in self.nodes)

    def leaf_ids(self, X, limit=None):
        out = np.zeros(len(X), dtype=int)

        def go(nid, rows):
            nd = self.nodes[nid]
            if nd.is_leaf or len(rows) == 0 or (limit is not None and nd.depth >= limit):
                out[rows] = nid
                return
            m = X[rows, nd.feature] <= nd.threshold
            go(nd.left, rows[m])
            go(nd.right, rows[~m])

        go(0, np.arange(len(X)))
        return out

    def predict(self, X, limit=None):
        return np.array([self.nodes[i].pred for i in self.leaf_ids(X, limit)])

    def importances(self):
        imp = np.zeros(self.n_features)
        for nd in self.nodes:
            if not nd.is_leaf:
                imp[nd.feature] += nd.n * nd.gain / self.n_train
        s = imp.sum()
        return imp / s if s > 0 else imp

    def trace(self, x):
        path, steps, nd = [0], [], self.nodes[0]
        while not nd.is_leaf:
            went_left = x[nd.feature] <= nd.threshold
            steps.append((nd, went_left))
            nd = self.nodes[nd.left if went_left else nd.right]
            path.append(nd.id)
        return path, steps


@st.cache_data(show_spinner=False)
def fit_tree(X, y, k, criterion, max_depth, min_split, min_leaf, min_gain) -> Tree:
    Y = np.eye(k)[y]
    N, P = X.shape
    nodes: list[Node] = []

    def grow(idx, depth):
        counts = Y[idx].sum(0)
        imp = float(impurity(counts[None], np.array([len(idx)]), criterion)[0])
        node = Node(len(nodes), depth, idx, counts, imp)
        nodes.append(node)
        if depth >= max_depth or len(idx) < min_split or imp <= 1e-12 or len(idx) < 2 * min_leaf:
            return node.id
        best_g, best_j, best_t = -1.0, -1, np.nan
        Yn = Y[idx]
        for j in range(P):
            thr, g = scan_feature(X[idx, j], Yn, criterion, min_leaf)
            if len(g) and g.max() > best_g:
                i = int(np.argmax(g))
                best_g, best_j, best_t = float(g[i]), j, float(thr[i])
        if best_j < 0 or best_g <= 1e-12 or best_g * len(idx) / N < min_gain:
            return node.id
        node.feature, node.threshold, node.gain = best_j, best_t, best_g
        mask = X[idx, best_j] <= best_t
        node.left = grow(idx[mask], depth + 1)
        node.right = grow(idx[~mask], depth + 1)
        return node.id

    grow(np.arange(N), 0)
    return Tree(nodes, k, criterion, P, N)


# visualisation helpers
def tree_svg(tree, feats, names, colors, horizontal=False, limit=None,
             show_counts=True, show_imp=True, path=None):
    """Render the tree as a self-contained SVG (own layout → no JS/graphviz glitches).
    Every node paints its own opaque fill, so it is readable on light and dark themes."""
    k = len(names)
    pnodes = set(path or [])
    pedges = set(zip(path[:-1], path[1:])) if path else set()
    info, esc = {}, html.escape

    def lines_for(n, collapsed):
        if n.is_leaf:
            ls = [f"● {names[n.pred]}"]
        elif collapsed:
            ls = [f"⋯ subtree → {names[n.pred]}"]
        else:
            ls = [f"{feats[n.feature]} ≤ {n.threshold:.4g}"]
        if show_imp:
            ls.append(f"{tree.criterion} = {n.impurity:.3f}")
        ls.append(f"samples = {n.n}")
        if show_counts:
            v = "value = [" + ", ".join(str(int(c)) for c in n.counts) + "]"
            ls.append(v if len(v) <= 40 else v[:38] + "…]")
        if not n.is_leaf and not collapsed:
            ls.append(f"class = {names[n.pred]}")
        return ls

    def collect(nid):
        n = tree.nodes[nid]
        collapsed = limit is not None and n.depth >= limit and not n.is_leaf
        info[nid] = (lines_for(n, collapsed), collapsed)
        if not n.is_leaf and not collapsed:
            collect(n.left)
            collect(n.right)

    collect(0)
    max_chars = max(len(l) for ls, _ in info.values() for l in ls)
    max_lines = max(len(ls) for ls, _ in info.values())
    W = int(min(330, max(150, 6.6 * max_chars + 28)))
    H = 15 * max_lines + 38
    step_b = (H + 18) if horizontal else (W + 24)      # spacing across the tree
    step_d = (W + 76) if horizontal else (H + 58)      # spacing between levels

    pos, counter = {}, [0]

    def layout(nid):
        n = tree.nodes[nid]
        if n.is_leaf or info[nid][1]:
            b = counter[0]
            counter[0] += 1
        else:
            b = (layout(n.left) + layout(n.right)) / 2
        pos[nid] = (b, n.depth)
        return b

    layout(0)
    M = 16
    xy = lambda i: ((pos[i][1] * step_d, pos[i][0] * step_b) if horizontal
                    else (pos[i][0] * step_b, pos[i][1] * step_d))
    tw = int(max(xy(i)[0] for i in pos) + W + 2 * M)
    th = int(max(xy(i)[1] for i in pos) + H + 2 * M)

    edges, nodes = [], []
    for nid in pos:
        lines, collapsed = info[nid]
        n = tree.nodes[nid]
        x, y = xy(nid)
        x, y = x + M, y + M
        if not n.is_leaf and not collapsed:
            for child, lab in ((n.left, "yes"), (n.right, "no")):
                cx, cy = xy(child)
                cx, cy = cx + M, cy + M
                hot = (nid, child) in pedges
                col, sw = ("#f43f5e", 3) if hot else ("#94a3b8", 1.6)
                if horizontal:
                    x1, y1, x2, y2 = x + W, y + H / 2, cx, cy + H / 2
                    mx = (x1 + x2) / 2
                    d = f"M{x1},{y1} C{mx},{y1} {mx},{y2} {x2},{y2}"
                    lx, ly, anc = mx, (y1 + y2) / 2 + (-6 if lab == "yes" else 15), "middle"
                else:
                    x1, y1, x2, y2 = x + W / 2, y + H, cx + W / 2, cy
                    my = (y1 + y2) / 2
                    d = f"M{x1},{y1} C{x1},{my} {x2},{my} {x2},{y2}"
                    lx, ly = (x1 + x2) / 2 + (-8 if lab == "yes" else 8), my + 4
                    anc = "end" if lab == "yes" else "start"
                edges.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{sw}"/>'
                             f'<text x="{lx}" y="{ly}" text-anchor="{anc}" font-size="11" '
                             f'fill="{"#f43f5e" if hot else "#94a3b8"}">{lab}</text>')
        p = n.counts / n.counts.sum()
        strength = (p.max() - 1 / k) / (1 - 1 / k)
        fill = hex_mix(colors[n.pred % len(colors)], 0.22 + 0.6 * strength)
        stroke, sw = ("#f43f5e", 3.2) if nid in pnodes else ("#64748b", 1.3)
        rx = 18 if n.is_leaf else 6
        dash = ' stroke-dasharray="5 4"' if collapsed else ""
        g = [f'<rect x="{x}" y="{y}" width="{W}" height="{H}" rx="{rx}" fill="{fill}" '
             f'stroke="{stroke}" stroke-width="{sw}"{dash}/>']
        for li, line in enumerate(lines):
            wt = ' font-weight="700"' if li == 0 else ""
            g.append(f'<text x="{x + W / 2}" y="{y + 20 + li * 15}" text-anchor="middle" '
                     f'font-size="11.5" fill="#0f172a"{wt}>{esc(line)}</text>')
        bx, bw, acc = x + 14, W - 28, 0.0           # class-mix bar
        for ci in range(k):
            if n.counts[ci] > 0:
                w_ = bw * n.counts[ci] / n.n
                g.append(f'<rect x="{bx + acc:.2f}" y="{y + H - 14}" width="{w_:.2f}" height="6" '
                         f'fill="{colors[ci]}"/>')
                acc += w_
        nodes.append("".join(g))
    return "".join(edges) + "".join(nodes), tw, th


def wrap_svg(inner, w, h, style=""):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'font-family="Inter, Segoe UI, Helvetica, Arial, sans-serif" style="{style}">{inner}</svg>')


def show_tree(inner, w, h, fit=True, zoom=1.0):
    if fit:
        svg = wrap_svg(inner, w, h, "max-width:100%;height:auto;display:block;margin:auto")
        height = int(min(780, h * min(1.0, 1300 / w) + 24))
    else:
        svg = wrap_svg(inner, w, h, f"width:{w * zoom:.0f}px;height:{h * zoom:.0f}px;display:block")
        height = int(min(780, h * zoom + 24))
    page = ('<html><head><meta charset="utf-8"></head><body style="margin:0;background:transparent">'
            f'<div style="overflow:auto">{svg}</div></body></html>')
    height = max(height, 160)
    if hasattr(st, "iframe"):                      
        st.iframe(page, height=height)
    else:
        components.html(page, height=height, scrolling=True)

# Data loading / preparation
@st.cache_data(show_spinner=False)
def load_builtin(name: str) -> pd.DataFrame:
    from sklearn import datasets as ds
    loader = {"Iris": ds.load_iris, "Wine": ds.load_wine,
              "Breast cancer": ds.load_breast_cancer,
              "Digits (64 features)": ds.load_digits}[name]
    d = loader()
    df = pd.DataFrame(d.data, columns=[str(c).replace(" (cm)", "") for c in d.feature_names])
    df["target"] = [str(d.target_names[i]) for i in d.target]
    return df


@st.cache_data(show_spinner=False)
def make_synthetic(n, p, p_inf, k, sep, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    X = rng.normal(0, 1, (n, p))
    X[:, :p_inf] += rng.normal(0, sep, (k, p_inf))[y]
    perm = rng.permutation(p)
    X = X[:, perm]
    cols = [f"x{i + 1}" for i in range(p)]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = [f"Class {c + 1}" for c in y]
    informative = [cols[j] for j in range(p) if perm[j] < p_inf]
    return df, informative


def prepare_features(df, feats):
    X, notes = df[feats].copy(), []
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) or X[c].dtype == bool]
    cat = [c for c in X.columns if c not in num]
    parts = []
    if num:
        N = X[num].astype(float)
        if N.isna().any().any():
            notes.append("Missing numeric values were filled with the column median.")
            N = N.fillna(N.median())
        parts.append(N)
    for c in cat:
        if X[c].nunique() <= 20:
            parts.append(pd.get_dummies(X[c].astype(str), prefix=c, dtype=float))
            notes.append(f"`{c}` is categorical → one-hot encoded.")
        else:
            notes.append(f"`{c}` dropped (categorical with >20 levels).")
    return (pd.concat(parts, axis=1) if parts else pd.DataFrame()), notes


def strat_split(y, test_size, seed):
    rng = np.random.default_rng(seed)
    te = []
    for c in np.unique(y):
        ids = np.where(y == c)[0]
        rng.shuffle(ids)
        te.extend(ids[: int(round(len(ids) * test_size))])
    te = np.array(sorted(te), dtype=int)
    mask = np.ones(len(y), bool)
    mask[te] = False
    return np.where(mask)[0], te


def accuracy(a, b):
    return float((a == b).mean()) if len(a) else float("nan")


# Sidebar
with st.sidebar:
    st.markdown("## Controls")
    st.markdown("##### 1 · Data")
    source = st.selectbox("Dataset", ["Iris", "Wine", "Breast cancer", "Digits (64 features)",
                                      "Synthetic (n features)", "Upload CSV"])
    df, target, note_extra = None, "target", None
    if source == "Synthetic (n features)":
        s1, s2 = st.columns(2)
        n_samp = s1.number_input("Samples", 100, 20000, 800, step=100)
        n_cls = s2.number_input("Classes", 2, 8, 3)
        n_feat = st.slider("Total features", 2, 100, 12)
        n_inf = st.slider("Informative features", 1, n_feat, min(4, n_feat))
        sep = st.slider("Class separation", 0.2, 3.0, 1.2, 0.1)
        seed_d = st.number_input("Data seed", 0, 9999, 7)
        df, inf_cols = make_synthetic(int(n_samp), n_feat, n_inf, int(n_cls), sep, int(seed_d))
        note_extra = "Ground-truth informative features: " + ", ".join(inf_cols)
    elif source == "Upload CSV":
        up = st.file_uploader("CSV file", type="csv")
        if up is not None:
            df = pd.read_csv(up)
            target = st.selectbox("Target column", df.columns, index=len(df.columns) - 1)
    else:
        df = load_builtin(source)

    if df is not None:
        opts = [c for c in df.columns if c != target]
        with st.expander(f"Features ({len(opts)} available)", expanded=False):
            feats_sel = st.multiselect("Use features", opts, default=opts,
                                       key=f"feats_{source}_{target}_{len(opts)}",
                                       label_visibility="collapsed")
        if note_extra:
            st.caption(note_extra)

    st.markdown("##### 2 · Split criterion")
    crit_label = st.radio("Criterion", ["Entropy", "Gini"], horizontal=True,
                          label_visibility="collapsed")
    criterion = crit_label.lower()
    if criterion == "entropy":
        st.latex(r"H=-\sum_i p_i\log_2 p_i")
        st.caption("Gain = H(parent) − weighted H(children)")
    else:
        st.latex(r"G=1-\sum_i p_i^2")
        st.caption("Gain = G(parent) − weighted G(children)")

    st.markdown("##### 3 · Tree shape")
    max_depth = st.slider("Max depth", 1, 15, 4)
    min_split = st.slider("Min samples to split", 2, 60, 2)
    min_leaf = st.slider("Min samples per leaf", 1, 60, 1)
    min_gain = st.slider("Min weighted gain", 0.0, 0.10, 0.0, 0.001, format="%.3f",
                         help="A split must reduce impurity (weighted by node size) by at least this much.")

    st.markdown("##### 4 · Evaluation")
    test_size = st.slider("Test set size", 0.10, 0.50, 0.25, 0.05)
    split_seed = st.number_input("Split seed", 0, 9999, 42)

# Hero
def hero(pills):
    st.markdown(
        '<div class="hero"><h1>Decision Tree Simulator</h1>'
        + "".join(f'<span class="pill">{p}</span>' for p in pills) + "</div>",
        unsafe_allow_html=True)


if df is None:
    hero(["Upload a CSV to begin"])
    st.info("👈 Upload a CSV in the sidebar, pick the target column, and choose your features.")
    st.stop()

if not feats_sel:
    hero([source])
    st.warning("Select at least one feature in the sidebar.")
    st.stop()

X_df, notes = prepare_features(df, feats_sel)
if X_df.shape[1] == 0:
    st.error("No usable features after preprocessing.")
    st.stop()
codes, uniques = pd.factorize(df[target].astype(str), sort=True)
names = [str(u) for u in uniques]
k = len(names)
if k < 2:
    st.error("The target needs at least two classes.")
    st.stop()
colors = [PALETTE[i % len(PALETTE)] for i in range(k)]
feats = list(X_df.columns)
X = X_df.to_numpy(dtype=float)
y = codes.astype(int)
tr_i, te_i = strat_split(y, test_size, int(split_seed))
Xtr, ytr, Xte, yte = X[tr_i], y[tr_i], X[te_i], y[te_i]
sig = hashlib.md5(f"{source}|{len(df)}|{'|'.join(feats)}|{target}".encode()).hexdigest()[:8]

params = (k, criterion, max_depth, min_split, min_leaf, min_gain)
tree = fit_tree(Xtr, ytr, *params)
tr_acc = accuracy(tree.predict(Xtr), ytr)
te_acc = accuracy(tree.predict(Xte), yte)

hero([source, f"{len(df):,} samples", f"{len(feats)} features", f"{k} classes",
      f"criterion: {crit_label}"])
for msg in notes:
    st.caption("ℹ️ " + msg)

c = st.columns(5)
c[0].metric("Train accuracy", f"{tr_acc:.1%}")
c[1].metric("Test accuracy", f"{te_acc:.1%}", f"{(te_acc - tr_acc) * 100:+.1f} pts vs train",
            delta_color="off")
c[2].metric("Depth", tree.depth)
c[3].metric("Leaves", tree.n_leaves)
c[4].metric("Nodes", len(tree.nodes))
if tr_acc - te_acc > 0.10:
    st.caption(":red[Large train→test gap — the tree is likely overfitting. ]"
               " :orange[Try a lower max depth or higher min samples per leaf.]")

tabs = st.tabs(["Tree", "Split explorer", "Predict", "Performance", "Data"])

# Tab 1 — Tree
with tabs[0]:
    o1, o2, o3, o4, o5 = st.columns([1.3, 1.3, 1, 1, 1])
    horizontal = o1.radio("Layout", ["Top → down", "Left → right"], horizontal=True) == "Left → right"
    if tree.depth >= 2:
        disp = o2.slider("Depth shown", 1, tree.depth, min(tree.depth, 5),
                         help="Deeper subtrees are collapsed for readability.")
    else:
        disp = tree.depth
    show_counts = o3.toggle("Class counts", True)
    show_imp = o4.toggle("Impurity", True)
    fit = o5.toggle("Fit to width", True)
    zoom = 1.0 if fit else st.slider("Zoom", 30, 200, 100, 10, format="%d%%") / 100
    st.markdown(legend(names, colors), unsafe_allow_html=True)
    inner, tw, th = tree_svg(tree, feats, names, colors, horizontal, disp, show_counts, show_imp)
    show_tree(inner, tw, th, fit, zoom)
    
# Tab 2 — Split explorer
with tabs[1]:
    gain_name = "Information gain" if criterion == "entropy" else "Gini gain"
    
    node_label = lambda i: (f"#{i} · depth {tree.nodes[i].depth} · n={tree.nodes[i].n}"
                            + (" · leaf" if tree.nodes[i].is_leaf else
                               f" · {feats[tree.nodes[i].feature]} ≤ {tree.nodes[i].threshold:.3g}"))
    nid = st.selectbox("Choose a node to inspect", list(range(len(tree.nodes))),
                       format_func=node_label, key=f"node_{sig}")
    node = tree.nodes[nid]
    Xn, Yn = Xtr[node.idx], np.eye(k)[ytr[node.idx]]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Samples", node.n)
    m2.metric(f"Impurity ({criterion})", f"{node.impurity:.4f}")
    m3.metric("Majority class", names[node.pred])
    m4.metric(gain_name, f"{node.gain:.4f}" if not node.is_leaf else "—")
    st.markdown(dist_bar(node.counts, names, colors), unsafe_allow_html=True)
    st.markdown(f'<div class="small">{legend(names, colors)}</div>', unsafe_allow_html=True)

    if node.n < 2:
        st.info("This node has a single sample — nothing to split.")
    else:
        rows = []
        for j, f in enumerate(feats):
            thr, g = scan_feature(Xn[:, j], Yn, criterion, min_leaf)
            if len(g):
                i = int(np.argmax(g))
                rows.append((f, j, float(g[i]), float(thr[i])))
            else:
                rows.append((f, j, 0.0, float("nan")))
        res = pd.DataFrame(rows, columns=["Feature", "j", "Best gain", "Best threshold"])
        res["Chosen"] = np.where((res.j == node.feature) & (not node.is_leaf), "✓", "")
        res = res.sort_values("Best gain", ascending=False).reset_index(drop=True)

        left, right = st.columns([1.1, 1])
        with left:
            st.markdown(f"**Best {gain_name.lower()} per feature**")
            top = res.head(15).iloc[::-1]
            fig = go.Figure(go.Bar(
                x=top["Best gain"], y=top["Feature"], orientation="h",
                marker_color=["#6366F1" if c_ == "✓" else "rgba(148,163,184,.65)" for c_ in top["Chosen"]],
                hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
            wide(st.plotly_chart, style_fig(fig, max(240, 26 * len(top) + 50)))
        with right:
            st.markdown("**All features**")
            wide(st.dataframe, res.drop(columns="j"), hide_index=True, height=min(420, 36 * len(res) + 40))

# Tab 3 — Predict
with tabs[2]:
    def load_sample():
        i = int(np.random.randint(len(Xte)))
        for jj in range(len(feats)):
            st.session_state[f"in_{sig}_{jj}"] = float(Xte[i, jj])
        st.session_state[f"truth_{sig}"] = names[yte[i]]

    cin, cout = st.columns([1, 1.35], gap="large")
    with cin:
        h1, h2 = st.columns([2, 1])
        h1.markdown("**Feature values**")
        if len(Xte):
            h2.button("Random test row", on_click=load_sample)
        box = st.container(height=420, border=True) if len(feats) > 8 else st.container()
        xin = np.zeros(len(feats))
        with box:
            ncol = 2 if len(feats) > 4 else 1
            cols = st.columns(ncol)
            for j, f in enumerate(feats):
                key = f"in_{sig}_{j}"
                col = cols[j % ncol]
                vals = np.unique(Xtr[:, j])
                mn, mx = float(vals.min()), float(vals.max())
                st.session_state.setdefault(key, float(np.median(Xtr[:, j])))
                if len(vals) <= 2:
                    opts = [float(v) for v in vals]
                    if st.session_state[key] not in opts:
                        st.session_state[key] = opts[0]
                    xin[j] = col.selectbox(f, opts, key=key)
                else:
                    st.session_state[key] = float(np.clip(st.session_state[key], mn, mx))
                    xin[j] = col.slider(f, mn, mx, step=(mx - mn) / 200, format="%.4g", key=key)

    with cout:
        path, steps = tree.trace(xin)
        leaf = tree.nodes[path[-1]]
        st.markdown(
            f'<div class="card" style="border-color:{colors[leaf.pred]}">'
            f'<span class="small">PREDICTION</span><br>'
            f'<span style="font-size:1.7rem;font-weight:700;color:{colors[leaf.pred]}">{names[leaf.pred]}</span>'
            f'<div style="margin-top:.5rem">{dist_bar(leaf.counts, names, colors, 10)}</div></div>',
            unsafe_allow_html=True)
        st.markdown("**Decision path**")
        for s, (nd, went_left) in enumerate(steps, 1):
            op = "≤" if went_left else ">"
            st.markdown(f'<div class="step">{s}. <b>{feats[nd.feature]}</b> = {xin[nd.feature]:.4g} '
                        f'{op} {nd.threshold:.4g} → go <b>{"left" if went_left else "right"}</b></div>',
                        unsafe_allow_html=True)
        st.markdown(f'<div class="step leaf">Reached leaf <b>{names[leaf.pred]}</b></div>',
                    unsafe_allow_html=True)
    with st.expander("Highlighted Path leads to the prediction", expanded=True):
        pin, pw, ph = tree_svg(tree, feats, names, colors, False, None, True, True, path)
        show_tree(pin, pw, ph, True)

# Tab 4 — Performance
with tabs[3]:
    p1, p2 = st.columns(2, gap="large")
    with p1:
        st.markdown("**Confusion matrix (test set)**")
        if len(Xte):
            pred = tree.predict(Xte)
            cm = np.zeros((k, k), int)
            np.add.at(cm, (yte, pred), 1)
            fig = go.Figure(go.Heatmap(z=cm, x=names, y=names, text=cm, texttemplate="%{text}",
                                       colorscale="Purples", showscale=False))
            fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual", yaxis_autorange="reversed")
            wide(st.plotly_chart, style_fig(fig, 340))
            tp = np.diag(cm).astype(float)
            prec = tp / np.maximum(cm.sum(0), 1)
            rec = tp / np.maximum(cm.sum(1), 1)
            f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
            rep = pd.DataFrame({"Class": names, "Precision": prec, "Recall": rec,
                                "F1": f1, "Support": cm.sum(1)})
            wide(st.dataframe, rep.style.format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1": "{:.2f}"}),
                 hide_index=True)
    with p2:
        st.markdown("**Feature importance**")
        imp = tree.importances()
        order = np.argsort(imp)[::-1][:20][::-1]
        fig = go.Figure(go.Bar(x=imp[order], y=[feats[i] for i in order], orientation="h",
                               marker_color="#6366F1"))
        wide(st.plotly_chart, style_fig(fig, max(260, 24 * len(order) + 60)))
        unused = int((imp == 0).sum())
        if unused:
            st.caption(f"{unused} of {len(feats)} features are not used by the tree.")

    st.markdown("**Accuracy vs. depth**")
    deep = fit_tree(Xtr, ytr, k, criterion, 15, min_split, min_leaf, min_gain)
    ds = list(range(1, max(deep.depth, 1) + 1))
    fig = go.Figure()
    fig.add_scatter(x=ds, y=[accuracy(deep.predict(Xtr, d), ytr) for d in ds], name="Train",
                    mode="lines+markers", line=dict(color="#6366F1", width=3))
    if len(Xte):
        fig.add_scatter(x=ds, y=[accuracy(deep.predict(Xte, d), yte) for d in ds], name="Test",
                        mode="lines+markers", line=dict(color="#F59E0B", width=3))
    fig.add_vline(x=max_depth, line=dict(color="#f43f5e", dash="dash"),
                  annotation_text="current", annotation_position="top")
    fig.update_layout(xaxis_title="max depth", yaxis_title="accuracy", yaxis_tickformat=".0%")
    wide(st.plotly_chart, style_fig(fig, 300))

    st.markdown("**Entropy vs Gini — same settings, side by side**")
    cmp = []
    for cr in ("entropy", "gini"):
        t_ = fit_tree(Xtr, ytr, k, cr, max_depth, min_split, min_leaf, min_gain)
        cmp.append({"Criterion": cr.title() + (" (current)" if cr == criterion else ""),
                    "Train acc": accuracy(t_.predict(Xtr), ytr),
                    "Test acc": accuracy(t_.predict(Xte), yte),
                    "Depth": t_.depth, "Leaves": t_.n_leaves,
                    "Root split": f"{feats[t_.nodes[0].feature]} ≤ {t_.nodes[0].threshold:.3g}"
                    if not t_.nodes[0].is_leaf else "—"})
    wide(st.dataframe, pd.DataFrame(cmp).style.format({"Train acc": "{:.1%}", "Test acc": "{:.1%}"}),
         hide_index=True)

# Tab 5 — Data
with tabs[4]:
    d1, d2 = st.columns([2, 1], gap="large")
    with d1:
        st.markdown(f"**Preview** · {len(df):,} rows × {len(feats)} model features")
        wide(st.dataframe, df.head(200), height=360)
    with d2:
        st.markdown("**Class balance**")
        vc = np.bincount(y, minlength=k)
        fig = go.Figure(go.Bar(x=names, y=vc, marker_color=colors))
        wide(st.plotly_chart, style_fig(fig, 300))
    st.markdown("**Feature summary**")
    wide(st.dataframe, X_df.describe().T, height=300)
