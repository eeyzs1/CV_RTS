# CV_TRY

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

pugd 400 : 0.9353
pugdr_cos_3_0.1_400 : 0.9390000000000001
pugdr_sin_3_0.1_400 : 0.9375
pugdw_cos_5_0.5_400 : 0.9333
pugdw_sin_5_0.5_400 : 0.9331



