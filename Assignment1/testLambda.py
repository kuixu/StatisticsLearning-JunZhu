import IRLS

x_arr,y_arr=IRLS.loadDotData("a9a/a9a")
x_test,y_test=IRLS.loadDotData("a9a/a9a.t")

for lam in range(1,100,5):
    #print lam*0.001
    trainHistoryL2=IRLS.train(x_arr,y_arr,x_test,y_test,regularizer="L2",w_lambda= lam*0.000001)
    IRLS.test(x_test,y_test,regularizer="L2")
