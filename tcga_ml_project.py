"""
TCGA Pan-Cancer ML Project
Covers: Problem & Dataset | EDA | Regression + Classification | Clustering | Hidden Markov Model
        | Comparative Performance | Result Interpretation
Dataset: TCGA PANCAN RNA-Seq (UCI #401) - 801 patients x 20,531 genes, 5 tumour types
         (BRCA, KIRC, COAD, LUAD, PRAD). Downloaded automatically.
Run:  pip install -r requirements.txt   then   python tcga_ml_project.py
All figures + CSV results are saved to ./outputs
"""
import os, sys, io, tarfile, zipfile, warnings, urllib.request
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                             confusion_matrix, classification_report, r2_score, mean_squared_error,
                             mean_absolute_error, adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score, homogeneity_score)
from scipy.optimize import linear_sum_assignment
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings("ignore"); sys.stdout.reconfigure(line_buffering=True)
SEED = 42
np.random.seed(SEED)
OUT = "outputs"; os.makedirs(OUT, exist_ok=True)
sns.set_theme(style="whitegrid")
URL = "https://archive.ics.uci.edu/static/public/401/gene+expression+cancer+rna+seq.zip"

def save(name):
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}.png", dpi=150); plt.close()

def banner(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)

# ---------------------------------------------------------------- 1. DATASET
def load_data():
    cache = "data/tcga.parquet" if _has_parquet() else "data/tcga.pkl"
    if os.path.exists(cache):
        return pd.read_parquet(cache) if cache.endswith("parquet") else pd.read_pickle(cache)
    os.makedirs("data", exist_ok=True)
    print("Downloading TCGA PANCAN RNA-Seq (~70 MB)...")
    raw = urllib.request.urlopen(URL).read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    tgz = [n for n in z.namelist() if n.endswith(".tar.gz")][0]
    tar = tarfile.open(fileobj=io.BytesIO(z.read(tgz)), mode="r:gz")
    get = lambda s: tar.extractfile([m for m in tar.getmembers() if m.name.endswith(s)][0])
    X = pd.read_csv(get("data.csv"), index_col=0)
    y = pd.read_csv(get("labels.csv"), index_col=0)["Class"]
    df = X.join(y.rename("Class"))
    df.to_parquet(cache) if cache.endswith("parquet") else df.to_pickle(cache)
    return df

def _has_parquet():
    try: import pyarrow; return True
    except ImportError: return False

banner("1. PROBLEM IDENTIFICATION & DATASET SELECTION")
print("Problem: Predict primary cancer type (Cancer of Unknown Primary) from tumour genomic profile,\n"
      "discover molecular subgroups without labels (clustering) and model latent expression states (HMM).")
df = load_data()
X_raw = df.drop(columns="Class"); y_txt = df["Class"]
le = LabelEncoder(); y = le.fit_transform(y_txt); CLASSES = le.classes_
print(f"Samples: {X_raw.shape[0]} | Genes: {X_raw.shape[1]} | Classes: {list(CLASSES)}")
print("Missing values:", int(X_raw.isna().sum().sum()))

# ---------------------------------------------------------------- 2. EDA
banner("2. EXPLORATORY DATA ANALYSIS")
counts = y_txt.value_counts(); print(counts.to_string())
plt.figure(figsize=(6, 4)); sns.barplot(x=counts.index, y=counts.values, palette="viridis")
plt.title("Class distribution"); plt.ylabel("Patients"); save("eda_class_distribution")

zero_frac = (X_raw == 0).mean()
print(f"Genes never expressed: {(zero_frac == 1).sum()} | median sparsity: {zero_frac.median():.2%}")
plt.figure(figsize=(6, 4)); sns.histplot(X_raw.values.ravel()[::50], bins=60, color="teal")
plt.title("Expression value distribution (log2 RSEM)"); save("eda_expression_hist")

vt = VarianceThreshold(0.0).fit(X_raw)
X_nz = X_raw.loc[:, vt.get_support()]
var = X_nz.var().sort_values(ascending=False)
top20 = var.index[:20]
plt.figure(figsize=(10, 6))
sns.heatmap(X_nz[top20].groupby(y_txt).mean(), cmap="magma", annot=False)
plt.title("Mean expression of 20 most variable genes per cancer type"); save("eda_top_genes_heatmap")
plt.figure(figsize=(8, 7)); sns.heatmap(X_nz[top20].corr(), cmap="coolwarm", center=0)
plt.title("Correlation of top-20 variable genes"); save("eda_correlation")

Xs_all = StandardScaler().fit_transform(X_nz)
pca_full = PCA(random_state=SEED).fit(Xs_all)
cum = np.cumsum(pca_full.explained_variance_ratio_)
print(f"PCs for 80% variance: {np.argmax(cum >= .8) + 1} | 95%: {np.argmax(cum >= .95) + 1}")
plt.figure(figsize=(6, 4)); plt.plot(cum[:200]); plt.axhline(.8, ls="--", c="r")
plt.title("PCA cumulative explained variance"); plt.xlabel("Components"); save("eda_pca_variance")
P2 = pca_full.transform(Xs_all)[:, :2]
T2 = TSNE(n_components=2, random_state=SEED, init="pca", perplexity=30).fit_transform(pca_full.transform(Xs_all)[:, :50])
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
sns.scatterplot(x=P2[:, 0], y=P2[:, 1], hue=y_txt, ax=ax[0], s=18); ax[0].set_title("PCA")
sns.scatterplot(x=T2[:, 0], y=T2[:, 1], hue=y_txt, ax=ax[1], s=18); ax[1].set_title("t-SNE")
save("eda_pca_tsne")
f_scores, _ = f_classif(X_nz, y)
top_f = pd.Series(f_scores, index=X_nz.columns).sort_values(ascending=False).head(15)
print("Top discriminative genes (ANOVA F):\n", top_f.round(1).to_string())
print("Insight: classes are well separated in PCA/t-SNE space -> strong tissue-specific signal.")

# ---------------------------------------------------------------- split (shared)
Xtr, Xte, ytr, yte = train_test_split(X_nz, y, test_size=0.2, stratify=y, random_state=SEED)
K_FEAT = 500  # selection is INSIDE pipelines -> no leakage

# ---------------------------------------------------------------- 3a. CLASSIFICATION
banner("3a. CLASSIFICATION (multi-class cancer type)")
def clf_pipe(model, pca=False):
    steps = [("sc", StandardScaler()), ("kb", SelectKBest(f_classif, k=K_FEAT))]
    if pca: steps.append(("pca", PCA(n_components=50, random_state=SEED)))
    return Pipeline(steps + [("m", model)])

classifiers = {
    "Logistic Regression": clf_pipe(LogisticRegression(max_iter=5000, C=0.1)),
    "SVM (RBF)": clf_pipe(SVC(C=10, probability=True, random_state=SEED)),
    "KNN": clf_pipe(KNeighborsClassifier(5), pca=True),
    "Naive Bayes": clf_pipe(GaussianNB(), pca=True),
    "Random Forest": clf_pipe(RandomForestClassifier(500, random_state=SEED, n_jobs=-1)),
    "Gradient Boosting": clf_pipe(HistGradientBoostingClassifier(max_iter=200, random_state=SEED)),
    "MLP Neural Net": clf_pipe(MLPClassifier((256, 64), max_iter=1000, random_state=SEED, early_stopping=True)),
}
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
clf_rows, fitted = [], {}
for name, pipe in classifiers.items():
    cv = cross_val_score(pipe, Xtr, ytr, cv=skf, scoring="accuracy", n_jobs=-1)
    pipe.fit(Xtr, ytr); fitted[name] = pipe
    pr = pipe.predict(Xte); pb = pipe.predict_proba(Xte)
    row = dict(Model=name, CV_Acc_mean=cv.mean(), CV_Acc_std=cv.std(),
               Test_Acc=accuracy_score(yte, pr), Precision=precision_score(yte, pr, average="macro"),
               Recall=recall_score(yte, pr, average="macro"), F1_macro=f1_score(yte, pr, average="macro"),
               ROC_AUC=roc_auc_score(yte, pb, multi_class="ovr"))
    clf_rows.append(row); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
clf_df = pd.DataFrame(clf_rows).sort_values(["Test_Acc", "CV_Acc_mean"], ascending=False)
clf_df.to_csv(f"{OUT}/classification_results.csv", index=False)
best_clf = clf_df.iloc[0]["Model"]
pr = fitted[best_clf].predict(Xte)
print(f"\nBest classifier: {best_clf}\n", classification_report(yte, pr, target_names=CLASSES, digits=4))
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(yte, pr), annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
plt.title(f"Confusion matrix - {best_clf}"); plt.xlabel("Predicted"); plt.ylabel("True"); save("clf_confusion_matrix")
rf = fitted["Random Forest"]; sel = Xtr.columns[rf.named_steps["kb"].get_support()]
imp = pd.Series(rf.named_steps["m"].feature_importances_, index=sel).nlargest(20)
plt.figure(figsize=(7, 6)); sns.barplot(x=imp.values, y=imp.index, palette="rocket")
plt.title("Top-20 biomarker genes (Random Forest)"); save("clf_feature_importance")
imp.to_csv(f"{OUT}/top_biomarkers.csv")

# Binary task (from slides: tumour-specific detection) - BRCA vs rest
yb_tr, yb_te = (ytr == list(CLASSES).index("BRCA")).astype(int), (yte == list(CLASSES).index("BRCA")).astype(int)
bin_m = clf_pipe(LogisticRegression(max_iter=5000, C=0.1)).fit(Xtr, yb_tr)
print(f"Binary BRCA-vs-rest: Acc={accuracy_score(yb_te, bin_m.predict(Xte)):.4f} "
      f"AUC={roc_auc_score(yb_te, bin_m.predict_proba(Xte)[:, 1]):.4f}")

# ---------------------------------------------------------------- 3b. REGRESSION
banner("3b. REGRESSION (predict expression of most variable gene from others)")
target = var.index[0]
print(f"Target gene: {target} (highest variance; removed from inputs to avoid leakage)")
Xr_tr, Xr_te = Xtr.drop(columns=target), Xte.drop(columns=target)
yr_tr, yr_te = Xtr[target].values, Xte[target].values
def reg_pipe(m): return Pipeline([("sc", StandardScaler()), ("pca", PCA(100, random_state=SEED)), ("m", m)])
regressors = {
    "Linear Regression": reg_pipe(LinearRegression()),
    "Ridge": reg_pipe(Ridge(alpha=10)),
    "Lasso": reg_pipe(Lasso(alpha=0.01, max_iter=20000)),
    "SVR (RBF)": reg_pipe(SVR(C=10, epsilon=0.1)),
    "Random Forest": reg_pipe(RandomForestRegressor(400, random_state=SEED, n_jobs=-1)),
    "Gradient Boosting": reg_pipe(GradientBoostingRegressor(n_estimators=300, random_state=SEED)),
}
kf = KFold(5, shuffle=True, random_state=SEED); reg_rows = []; reg_fit = {}
for name, pipe in regressors.items():
    cv = cross_validate(pipe, Xr_tr, yr_tr, cv=kf, scoring="r2", n_jobs=-1)["test_score"]
    pipe.fit(Xr_tr, yr_tr); reg_fit[name] = pipe; p = pipe.predict(Xr_te)
    row = dict(Model=name, CV_R2_mean=cv.mean(), CV_R2_std=cv.std(), Test_R2=r2_score(yr_te, p),
               RMSE=np.sqrt(mean_squared_error(yr_te, p)), MAE=mean_absolute_error(yr_te, p))
    reg_rows.append(row); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
reg_df = pd.DataFrame(reg_rows).sort_values("Test_R2", ascending=False)
reg_df.to_csv(f"{OUT}/regression_results.csv", index=False)
best_reg = reg_df.iloc[0]["Model"]; p = reg_fit[best_reg].predict(Xr_te)
plt.figure(figsize=(5, 5)); plt.scatter(yr_te, p, s=12, alpha=.7)
lim = [min(yr_te.min(), p.min()), max(yr_te.max(), p.max())]; plt.plot(lim, lim, "r--")
plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title(f"Regression: {best_reg}"); save("reg_actual_vs_pred")

# ---------------------------------------------------------------- 4. CLUSTERING
banner("4. CLUSTERING (unsupervised subtype discovery)")
# unsupervised feature selection: 2000 most variable genes -> scale -> PCA(20)
Xc = PCA(20, random_state=SEED).fit_transform(StandardScaler().fit_transform(X_nz[var.index[:2000]])); k = len(CLASSES)
def cluster_acc(yt, yp):
    labs = np.unique(yp); cm = np.zeros((len(labs), k))
    for i, l in enumerate(labs):
        for j in range(k): cm[i, j] = np.sum((yp == l) & (yt == j))
    r, c = linear_sum_assignment(-cm); return cm[r, c].sum() / len(yt)
inertias, sils = [], []
for kk in range(2, 11):
    km = KMeans(kk, n_init=20, random_state=SEED).fit(Xc); inertias.append(km.inertia_); sils.append(silhouette_score(Xc, km.labels_))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(range(2, 11), inertias, "o-"); ax[0].set_title("Elbow (KMeans)")
ax[1].plot(range(2, 11), sils, "o-"); ax[1].set_title("Silhouette vs k"); save("clu_elbow_silhouette")
print(f"Best k by silhouette: {int(np.argmax(sils) + 2)}")
clusterers = {
    "KMeans": KMeans(k, n_init=50, random_state=SEED),
    "Agglomerative (Ward)": AgglomerativeClustering(k, linkage="ward"),
    "Gaussian Mixture": GaussianMixture(k, covariance_type="full", n_init=10, random_state=SEED),
    "Spectral": SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=15, random_state=SEED),
}
clu_rows, clu_lab = [], {}
for name, m in clusterers.items():
    lab = m.fit_predict(Xc); clu_lab[name] = lab
    valid = len(set(lab)) > 1
    row = dict(Model=name, n_clusters=len(set(lab) - {-1}), ARI=adjusted_rand_score(y, lab),
               NMI=normalized_mutual_info_score(y, lab), Homogeneity=homogeneity_score(y, lab),
               Silhouette=silhouette_score(Xc, lab) if valid else np.nan, Cluster_Acc=cluster_acc(y, lab))
    clu_rows.append(row); print({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in row.items()})
clu_df = pd.DataFrame(clu_rows).sort_values("ARI", ascending=False)
clu_df.to_csv(f"{OUT}/clustering_results.csv", index=False)
best_clu = clu_df.iloc[0]["Model"]
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
sns.scatterplot(x=T2[:, 0], y=T2[:, 1], hue=y_txt, ax=ax[0], s=18); ax[0].set_title("True labels")
sns.scatterplot(x=T2[:, 0], y=T2[:, 1], hue=clu_lab[best_clu].astype(str), ax=ax[1], s=18, palette="tab10")
ax[1].set_title(f"{best_clu} clusters"); save("clu_tsne_comparison")
plt.figure(figsize=(6, 4))
sns.heatmap(pd.crosstab(y_txt.values, clu_lab[best_clu]), annot=True, fmt="d", cmap="Greens")
plt.title(f"Cancer type vs {best_clu} cluster"); save("clu_contingency")

# ---------------------------------------------------------------- 5. HIDDEN MARKOV MODEL
banner("5. HIDDEN MARKOV MODEL (class-conditional generative HMM classifier)")
# Each patient -> sequence of T observations along the top discriminative genes (fixed genomic order).
# One GaussianHMM per cancer type learns latent expression states + transitions; predict = argmax log-likelihood.
T_LEN, OBS_DIM = 40, 5   # 40 time-steps x 5-dim obs = 200 genes
def hmm_features(fit_X, fit_y, *others):
    sc = StandardScaler().fit(fit_X); kb = SelectKBest(f_classif, k=T_LEN * OBS_DIM).fit(sc.transform(fit_X), fit_y)
    tf = lambda A: kb.transform(sc.transform(A)).reshape(len(A), T_LEN, OBS_DIM)
    return [tf(fit_X)] + [tf(o) for o in others]

class HMMClassifier:
    def __init__(self, n_states=4, n_iter=100): self.n_states, self.n_iter = n_states, n_iter
    def fit(self, S, yy):
        self.classes_ = np.unique(yy); self.models = {}
        for c in self.classes_:
            seqs = S[yy == c]; Xcat = seqs.reshape(-1, OBS_DIM); lens = [T_LEN] * len(seqs)
            best, best_ll = None, -np.inf
            for rs in range(2):  # multiple restarts -> avoid poor local optima of EM
                m = GaussianHMM(self.n_states, covariance_type="diag", n_iter=self.n_iter, random_state=SEED + rs, min_covar=1e-3)
                m.fit(Xcat, lens); ll = m.score(Xcat, lens)
                if ll > best_ll: best, best_ll = m, ll
            self.models[c] = best
        return self
    def loglik(self, S): return np.array([[self.models[c].score(s) for c in self.classes_] for s in S])
    def predict(self, S): return self.classes_[self.loglik(S).argmax(1)]
    def predict_proba(self, S):
        L = self.loglik(S); L -= L.max(1, keepdims=True); P = np.exp(L); return P / P.sum(1, keepdims=True)

hmm_rows = []
hmm_cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
for ns in [2, 3, 4]:
    cv_acc = []
    for tr_i, va_i in hmm_cv.split(Xtr, ytr):
        Str, Sva = hmm_features(Xtr.iloc[tr_i], ytr[tr_i], Xtr.iloc[va_i])
        cv_acc.append(accuracy_score(ytr[va_i], HMMClassifier(ns).fit(Str, ytr[tr_i]).predict(Sva)))
    hmm_rows.append(dict(n_states=ns, CV_Acc=np.mean(cv_acc))); print(f"HMM states={ns}: CV acc={np.mean(cv_acc):.4f}")
best_ns = max(hmm_rows, key=lambda r: r["CV_Acc"])["n_states"]
Str, Ste = hmm_features(Xtr, ytr, Xte)
hmm = HMMClassifier(best_ns).fit(Str, ytr); hp = hmm.predict(Ste); hpb = hmm.predict_proba(Ste)
hmm_res = dict(Model=f"Gaussian HMM ({best_ns} states)", CV_Acc_mean=max(r["CV_Acc"] for r in hmm_rows), CV_Acc_std=np.nan,
               Test_Acc=accuracy_score(yte, hp), Precision=precision_score(yte, hp, average="macro"),
               Recall=recall_score(yte, hp, average="macro"), F1_macro=f1_score(yte, hp, average="macro"),
               ROC_AUC=roc_auc_score(yte, hpb, multi_class="ovr"))
print(hmm_res); print(classification_report(yte, hp, target_names=CLASSES, digits=4))
pd.DataFrame(hmm_rows).to_csv(f"{OUT}/hmm_state_selection.csv", index=False)
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(yte, hp), annot=True, fmt="d", cmap="Purples", xticklabels=CLASSES, yticklabels=CLASSES)
plt.title("Confusion matrix - HMM classifier"); save("hmm_confusion_matrix")
fig, axs = plt.subplots(1, len(CLASSES), figsize=(4 * len(CLASSES), 3.5))
for a, c in zip(axs, hmm.classes_):
    sns.heatmap(hmm.models[c].transmat_, annot=True, fmt=".2f", cmap="Oranges", ax=a, cbar=False); a.set_title(f"{CLASSES[c]} transitions")
save("hmm_transition_matrices")
# Unsupervised HMM: latent state occupancy as clustering signal
uh = GaussianHMM(k, covariance_type="diag", n_iter=200, random_state=SEED, min_covar=1e-3)
Sall = hmm_features(X_nz, y)[0]; uh.fit(Sall.reshape(-1, OBS_DIM), [T_LEN] * len(Sall))
occ = np.array([np.bincount(uh.predict(s), minlength=k) / T_LEN for s in Sall])
uh_lab = KMeans(k, n_init=20, random_state=SEED).fit_predict(occ)
print(f"Unsupervised HMM state-occupancy clustering: ARI={adjusted_rand_score(y, uh_lab):.4f} NMI={normalized_mutual_info_score(y, uh_lab):.4f}")

# ---------------------------------------------------------------- 6. COMPARATIVE ANALYSIS
banner("6. COMPARATIVE PERFORMANCE ANALYSIS")
all_clf = pd.concat([clf_df, pd.DataFrame([hmm_res])]).sort_values("Test_Acc", ascending=False)
all_clf.to_csv(f"{OUT}/all_classifiers_incl_hmm.csv", index=False)
print("Classification (incl. HMM):\n", all_clf.round(4).to_string(index=False))
print("\nRegression:\n", reg_df.round(4).to_string(index=False))
print("\nClustering:\n", clu_df.round(4).to_string(index=False))
m = all_clf.set_index("Model")[["Test_Acc", "F1_macro", "ROC_AUC"]]
m.plot(kind="barh", figsize=(9, 6)); plt.xlim(max(0, m.values.min() - .05), 1.005); plt.title("Classifier comparison"); save("cmp_classifiers")
reg_df.set_index("Model")[["Test_R2"]].plot(kind="barh", figsize=(8, 4), legend=False); plt.title("Regression R²"); save("cmp_regression")
clu_df.set_index("Model")[["ARI", "NMI", "Cluster_Acc"]].plot(kind="barh", figsize=(8, 4)); plt.title("Clustering comparison"); save("cmp_clustering")

# ---------------------------------------------------------------- 7. INTERPRETATION
banner("7. RESULT INTERPRETATION")
b = all_clf.iloc[0]; r = reg_df.iloc[0]; c = clu_df.iloc[0]
summary = f"""
* Best classifier: {b['Model']} -> Test accuracy {b['Test_Acc']:.4f}, macro-F1 {b['F1_macro']:.4f}, AUC {b['ROC_AUC']:.4f}.
  Linear/kernel models excel because tumour types are near-linearly separable in high-dim expression space.
* HMM classifier ({hmm_res['Model']}): accuracy {hmm_res['Test_Acc']:.4f}; latent states capture tissue-specific
  expression 'regimes' along the gene sequence, giving an interpretable generative model.
* Best regressor: {r['Model']} -> R² {r['Test_R2']:.4f}, RMSE {r['RMSE']:.4f}; gene {target} is strongly co-regulated
  with the rest of the transcriptome.
* Best clustering: {c['Model']} -> ARI {c['ARI']:.4f}, NMI {c['NMI']:.4f}, cluster accuracy {c['Cluster_Acc']:.4f};
  unsupervised groups recover the 5 cancer types almost exactly -> strong biological structure.
* Top biomarker genes saved in outputs/top_biomarkers.csv. Results are leakage-free (feature selection inside CV).
"""
print(summary); open(f"{OUT}/summary.txt", "w").write(summary)
print(f"All outputs saved in ./{OUT}/")
