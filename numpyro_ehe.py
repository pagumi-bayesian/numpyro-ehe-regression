# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

import jax
import jax.numpy as jnp
import jax.random as random

import numpyro

from ehe_regression import EHERegression
from ehe_regression import (
    generate_regression_data,
    results_different_outlier_rate,
    plot_different_results,
)

# %%
# Example usage of the function
n_samples = 100
alpha = 5
beta = 2

df = generate_regression_data(
    n_samples=n_samples,
    alpha=alpha,
    beta=np.array(beta).reshape(-1),
    sigma=10,
    p_outlier=0.05,
    outlier_mean=10,
    outlier_scale=10,
)

plt.scatter(df["x"], df["y"])
plt.show()

# %%
# 普通の線形回帰を試してみる
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(df[["x"]], df["y"])

print("estimated alpha : {:.2f}".format(lr.intercept_))
print("estimated beta : {:.2f}".format(lr.coef_[0]))

# 実際の回帰直線と、線形回帰で推定された回帰直線の可視化
df_plt = pd.DataFrame(np.linspace(0, 100, 100), columns=["x"])
df_plt["y_true"] = alpha + beta * df_plt["x"]
df_plt["y_lr"] = lr.predict(df_plt[["x"]])

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(df["x"], df["y"], color="black", label="obs")
ax.plot(df_plt["x"], df_plt["y_true"], color="blue", label="true")
ax.plot(df_plt["x"], df_plt["y_lr"], color="red", label="linear regression")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("True and estimated regression lines")
ax.legend()
fig.show()


# %%
ehe_reg = EHERegression(model_type="ehe")
df_ehe_prior_pred = ehe_reg.generate_obs_prediction(
    prediction_type="prior", x_predictor=jnp.array(df_plt[["x"]])
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(df["x"], df["y"], color="black", label="obs")
ax.plot(df_plt["x"], df_ehe_prior_pred["mean"], color="blue", label="mean")
ax.fill_between(
    df_plt["x"],
    df_ehe_prior_pred["lwr"],
    df_ehe_prior_pred["upr"],
    color="blue",
    alpha=0.2,
    label="interval",
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Prior predictive distribution")
ax.legend()
fig.show()

# %%
ehe_reg.fit(x=jnp.array(df[["x"]]), y=jnp.array(df["y"]), print=True)

# %%
ehe_reg.plot_trace()
plt.show()

# %%
df_ehe_posterior_pred = ehe_reg.generate_obs_prediction(
    prediction_type="posterior", x_predictor=jnp.array(df_plt[["x"]])
)

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(df["x"], df["y"], color="black", label="obs")
ax.plot(df_plt["x"], df_ehe_posterior_pred["mean"], color="blue", label="mean")
ax.fill_between(
    df_plt["x"],
    df_ehe_posterior_pred["lwr"],
    df_ehe_posterior_pred["upr"],
    color="blue",
    alpha=0.2,
    label="interval",
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Posterior predictive distribution", loc="left")
ax.legend()
fig.show()

# %%
# 各観測値が外れ値である事後確率を求める
outlier_prob = ehe_reg.outlier_prob()
df_outlier_flg = df.copy()
df_outlier_flg["outlier_status"] = [
    "outlier" if p > 0.9 else "standard" for p in outlier_prob
]

fig, ax = plt.subplots(figsize=(7, 4))
sns.scatterplot(data=df_outlier_flg, x="x", y="y", hue="outlier_status", ax=ax)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
fig.show()

# %%
true_alpha = 5
true_beta = [0.0, 0.4, -1.5, 1.2, -2.4]

df_all_results = results_different_outlier_rate(
    candidate_p_outlier=[0.05, 0.1, 0.2, 0.3],
    true_alpha=true_alpha,
    true_beta=true_beta,
)

# %%
fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=False, constrained_layout=True)
axes = axes.flatten()
plot_different_results(
    df_all_results, true_alpha=true_alpha, true_beta=true_beta, axes=axes
)
fig.show()

# %%
