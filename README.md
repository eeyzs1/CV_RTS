# CV_RTS

3d loss landscape
训练模型收敛后用theta = θ(α,β)=θ ∗+αd1+βd2
针对相同的x,y去画loss lanscape图，
​loss lanscape的x是d1,y是d2,z是loss



使用train,valid的每个epoch的batch loss的方差去画2d loss landscape，以及总loss每个epoch的折线图
画test的batch loss的方差的条形图？



当前实验：
1. 调整epsilon范围
2. 尝试改成sam法，走高阶，原有pugd思路改造，不乘abs(p)

epoch数值低的数据是在我的电脑上训练的，高epoch的数据是在kaggle上训练的，之前没注意导致相同模型的训练结果数据不一致
从400开始固定模型及将test数据作为valid，train不再分割
按照新分割法重新训练结果

timing:按1/5分割，或者valid的acc下降时, 或者acc变化率低于某个值时

cifar-10:
pugd 400 : 0.9353
pugdrs_ : 
Radius:
pugdr_icos_2_0.01_400 : 0.9419000000000001
pugdr_isin_2_0.01_400 : 0.9417000000000001
pugdr_cos_1.5_0.5_400 : 0.9377000000000001
pugdr_sin_2.0_0.0_400 : 0.9395

Scale:
pugds_icos_2_1_400 : 0.9392
pugds_isin_3_0.8_400 : 0.9373
pugds_cos_1.5_0.0_400 : 0.9399000000000001
pugds_sin_2.0_0.0_400 : 0.9368



Timing: 100 epochs(0-99)
pugd : 0.9211
pugdt_delta_100xi10.0mu3_t3 : 9226000000000001, epochs-20(21)
pugdt_var_100init_t10.0gamma0.2_k10 : 0.925, epochs-11(12)


cifar-100:
pugd 400 : 0.7223
pugdrs_ : 
Radius:
pugdr_icos_1_0.3_400 : 0.7256
pugdr_isin_2_0.01_400 : 0.7315
pugdr_cos_3.0_0.0_400 : 0.7234
pugdr_sin_2.0_0.0_400 : 0.7233

Scale:
pugds_icos_1.0_0.01_400 : 0.7238
pugds_isin_2_0.1_400 : 0.7292000000000001
pugds_cos_3.0_1.5_400 : 0.7264
pugds_sin_2.0_1.4_400 : 0.7255

Radius and scale:
pugdrs_isin2_0.01_isin2.0_0.1_400 : 0.7248 
pugdrs_cos2_0_cos3.0_1.5_400 : 0.7231000000000001

export HF_HOME=/your/custom/path  # Linux/macOS 
set HF_HOME=D:\your\custom\path   # Windows 
from transformers import TRANSFORMERS_CACHE 
print(TRANSFORMERS_CACHE)


Theorem：定理。是文章中重要的数学化的论述，一般有严格的数学证明。

Proposition：可以翻译为命题，经过证明且interesting，但没有Theorem重要，比较常用。


Lemma：一种比较小的定理，通常lemma的提出是为了来逐步辅助证明Theorem，有时候可以将Theorem拆分成多个小的Lemma来逐步证明，以使得证明的思路更加清晰。很少情况下Lemma会以其自身的形式存在。

Corollary：推论，由Theorem推出来的结论，通常我们会直接说this is a corollary of Theorem A。

Property：性质，结果值得一记，但是没有Theorem深刻。

Claim：陈述，先论述然后会在后面进行论证，可以看作非正式的lemma。

Note：就是注解。

Remark：涉及到一些结论，相对而言，Note像是说明，而Remark则是非正式的定理。

Conjecture：猜测。一个未经证明的论述，但是被认为是真。

Axiom/Postulate：公理。不需要证明的论述，是所有其他Theorem的基础。