# CV_RTS

<!-- using git lfs tracking lfs, so need 
git clone url
then:
git lfs pull -->

3d loss landscape
训练模型收敛后用theta = θ(α,β)=θ ∗+αd1+βd2
针对相同的x,y去画loss lanscape图，
​loss lanscape的x是d1,y是d2,z是loss


# cifar-10:
pugd 400 : 0.9353
## Radius:
pugdr_icos_2_0.01_400 : 0.9419000000000001
pugdr_isin_2_0.01_400 : 0.9417000000000001
pugdr_cos_1.5_0.5_400 : 0.9377000000000001
pugdr_sin_2.0_0.0_400 : 0.9395

## Scale:
pugds_icos_2_1_400 : 0.9392
pugds_isin_3_0.8_400 : 0.9373
pugds_cos_1.5_0.0_400 : 0.9399000000000001
pugds_sin_2.0_0.0_400 : 0.9368

## Radius and scale:

## Timing: 100 epochs(0-99)
pugd : 0.9211
pugdt_delta_100xi10.0mu3_t3 : 0.9226000000000001, epochs-20(21)
pugdt_var_100init_t10.0gamma0.2_k10 : 0.925, epochs-11(12)


# cifar-100:
pugd 400 : 0.7223
## Radius:
pugdr_icos_1_0.3_400 : 0.7256
pugdr_isin_2_0.01_400 : 0.7315
pugdr_cos_3.0_0.0_400 : 0.7234
pugdr_sin_2.0_0.0_400 : 0.7233

## Scale:
pugds_icos_1.0_0.01_400 : 0.7238
pugds_isin_2_0.1_400 : 0.7292000000000001
pugds_cos_3.0_1.5_400 : 0.7264
pugds_sin_2.0_1.4_400 : 0.7255

## Radius and scale:
pugdrs_isin2_0.01_isin2.0_0.1_400 : 0.7248 
pugdrs_cos2_0_cos3.0_1.5_400 : 0.7231000000000001


# Fine-tune:
## cifar-10:
resnet18_pugd_200 : 0.8613000000000001
resnet18_pugdr_isin1_0.5_200 : 0.8639

vit_pugd_200 : 0.9659000000000001
vit_pugdr_icos2_0.01_200 : 0.9669000000000001

deit_pugd_200 : 0.9459000000000001
deit_pugdr_icos2_0.01_200 : 0.9477000000000001

## cifar-100:
resnet18_pugd_200 : 0.603
resnet18_pugdr_cos2_0.0_200 : 0.6034

vit_pugd_200 : 0.8578
vit_pugdr_icos2.0_0.001_200 : 0.8703000000000001

deit_pugd_200 : 0.7926000000000001
deit_pugdr_icos3_2_200 : 0.7942
