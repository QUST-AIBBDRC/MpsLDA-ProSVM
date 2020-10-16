function W=UpdataW_r(newX,newY,MLC,lambda,g,g_num,eta)

parameter=['-s 3 -c ',num2str(lambda), ' -B -1'];

model = train(newY,newX,parameter,MLC,g,g_num,eta); 

W=(model.w)';