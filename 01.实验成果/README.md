# 实验成果

文件列表：

cPINNs报告.md

cPINNs报告.pdf(建议看pdf版，md格式显示bug)

2D/3D算例(正反问题).ipynb

Burgers.ipynb



---

忽略下面的公式~~
$$
\begin{aligned}&\mathcal{L}_{1}\left(\boldsymbol{\theta_1} ; \nu_1;\mathcal{T}_{\Omega_1 \times T},\mathcal{T}_{\Omega_1 \times \{t_0\}},\mathcal{T}_{\Gamma_1 \times T} \right)\\&\mathcal{L}_{I}\left(\boldsymbol{\theta_1} ;\ \boldsymbol{\theta_2} ;\nu_1,\nu_2;\mathcal{T}_{I \times T},\right) \\&\mathcal{L}_{2}\left(\boldsymbol{\theta_2} ; \nu_2;\mathcal{T}_{\Omega_2 \times T},\mathcal{T}_{\Omega_2 \times \{t_0\}},\mathcal{T}_{\Gamma_2 \times T} \right) \\\end{aligned}
$$


$(g_{1},g_{2},g_{\nu_1},g_{\nu2}) = \nabla_{\theta} \mathcal{L}_{total}(\theta_1,\theta_2;\nu_1,\nu_2...)$

$\theta_1,\theta_2,\nu_1,\nu_2= Adam/L-BFGS(\theta_1,\theta_2,\nu_1,\nu_2;g_1,g_2,g_{\nu_1},g_{\nu2})$