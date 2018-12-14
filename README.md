# Markowitz-Optimization
This is used for finding the optimal portfolio using Markowitz method.
Use your data which you can download from Yahoo Finance and make sure the Excel file you are using is CSV file.
Better to convert the data you are using to Normal data as this method assumes each stock data is Normally distributed. 
You can convert your data to Normal by using Box - Cox transformation.
The Math which is used here is optimizing w_transpose*Cov*w with constraints w_Transpose*1 = 1 and w_Transpose*mu = target return.
After doing the necessary Lagrange multipliers, I wrote the formulas directly.
Change 'number' variable to the number of stocks you are going to take.
Enter your desired target return in the 'targetreturn' variable.
Enter your weights manually after you get the weights from Markowitz or use Pandas instead of doing the manual way.
Non-Normal tests have been performed to check if your stock data or your return data is Normal or not.
Finally change the variable 'nsample' to the number of trading days your data had.
