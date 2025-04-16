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
