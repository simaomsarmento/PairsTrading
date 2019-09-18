# PairsTrading

## Thesis to obtain the Master of Science Degree in Electrical and Computer Engineering

**September 2019**

**Simão Moraes Sarmento, simao.moraes.sarmento@tecnico.ulisboa.pt**

## Thesis Abstract:
Pairs Trading is one of the most valuable market-neutral strategies used by hedge funds. It is particularly interesting as it overcomes the arduous process of valuing securities by focusing on relative pricing. By buying a relatively undervalued security and selling a relatively overvalued one, a profit can be made upon the pair's price convergence. However, with the growing availability of data, it became increasingly harder to find rewarding pairs. In this work, we address two problems: (i) how to find profitable pairs while constraining the search space and (ii) how to avoid long decline periods due to prolonged divergent pairs. To manage these difficulties, the application of promising Machine Learning techniques is investigated in detail. We propose the integration of an Unsupervised Learning algorithm, OPTICS, to handle problem (i). The results obtained demonstrate the suggested technique can outperform the common pairs' search methods, achieving an average portfolio Sharpe ratio of 3.79, in comparison to 3.58 and 2.59 obtained by standard approaches. For problem (ii), we introduce a forecasting-based trading model, capable of reducing the periods of portfolio decline by 75\%. Yet, this comes at the expense of decreasing overall profitability. The proposed strategy is tested using an ARMA model, an LSTM and an LSTM Encoder-Decoder. This work's results are simulated during varying periods between January 2009 and December 2018, using 5-minutes price data from a group of 208 commodity-linked ETFs, and accounting for transaction costs.  

## Repository content:

This repository contains all the code developed to produce the results presented in *Thesis.pdf*.

A detailed explanation concerning the code organization can be found in *code_organization.pdf*.





## Notes:

- The files have been organized in folders to make this repo tidier. Nevertheless, the code presented in the notebooks and 
in the training files presumes the class files are in the same directory. 
- To rerun the notebooks or the training files the path to the classes must be adapted.
 
Data available in: https://www.dropbox.com/sh/0w3vu1eylrfnkch/AABttIlDf64MmVf5CP1Qy-XOa?dl=0



## Training the Deep Learning models on Google Colab

1. Copy all the required files (data folder + classes + training files) to directory in Google Drive
2. Run the notebook in the 'training' folder using google colab.

