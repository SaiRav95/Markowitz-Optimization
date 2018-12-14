# Markowitz-Optimization
This is used for finding the optimal portfolio using Markowitz method.
Use your data which you can download from Yahoo Finance and make sure the Excel file you are using is CSV file.
Better to convert the data you are using to Normal data as this method assumes each stock data is Normally distributed. 
You can convert your data to Normal by using Box - Cox transformation.
The Math which is used here is optimizing w_transpose*Cov*w with constraints w_Transpose*1 = 1 and w_Transpose*mu = target return.
After doing the necessary Lagrange multipliers, I wrote the formulas directly.
