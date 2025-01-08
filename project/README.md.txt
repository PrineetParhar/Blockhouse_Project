README

Description of the Task

The goal of this task is to process high-frequency equity market data to compute the Order Flow Imbalance on five levels of the Limit Order Book of five different stocks. Additionally, we must evaluate cross-asset impacts on short-term price changes by reading the "Cross-Impact of Order Flow Imbalance in Equity Markets" paper. I have plotted my results on two graphs, both of which are shown in the results directory of this project.

How to Run

To run the analysis, you first must install the packages that are listed in the requirements.txt file. You must replace the Databento API key with your own. Then, you can run the python script visualizations.py to download the data, perform the requisite calculations, and visualize the data.

Brief Summary of Findings

Each stock's returns are regressed against the other stocks' OFI features.
These coefficients are small, showing not that much cross-impact between the five stocks shown. This could be for two reasons. Firstly, all five stocks are from different sectors of the market, so there is limited connection between them. Secondly, the data is being sampled at such a fast rate that only minimal change has the potential to occur.