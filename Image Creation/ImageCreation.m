files=dir('Non_Malicious_Class_Label_1\*.csv');
count = 0;
for file = files'
   fname=strcat('Non_Malicious_Class_Label_1\',file.name);
   m  = readtable(fname,'ReadVariableNames',false);
   TJNew= removevars(m,{'Var1'});
   TJNew = TJNew{:,:};
   originalImage = mat2gray(TJNew);
   I = imresize(originalImage, [32, 32]);
   count = count + 1;
   imagename = strcat('Non_Malicious_Class_Label_1_Image\\resize_img',num2str(count));
   imagename = strcat(imagename,'.jpg');
   imwrite(I,imagename,'jpg');
end