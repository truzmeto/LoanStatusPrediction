import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.plotly import iplot
import plotly.graph_objs as go
import plotly
plotly.tools.set_credentials_file(username='truzmeto', api_key='56BxCdCGkR66rnyKQMw7')


def plotLoanStat1(df, colors):
    """
    Plots pie plot -- fractions of good and bad loans
          bar plot -- loan amount vs time(month) sorted by loan status
    
    """
    
    f, ax = plt.subplots(1,2, figsize=(16,6))
    fs = 12
    plt.suptitle('Information on Loan Status', fontsize=fs)
    df["loan_status"].value_counts().plot.pie(explode=[0,0.1],
                                       autopct='%1.2f%%',
                                       ax=ax[0],
                                       shadow=True,
                                       colors=colors,
                                       fontsize=fs,
                                       startangle=175)

    ax[0].set_title('Loan Status', fontsize=fs)
    ax[0].set_ylabel('% of Loan by Status', fontsize=fs)
    
    sns.barplot(x="issue_month", y="loan_amnt", hue="loan_status",
                data=df, palette=colors,
                estimator=lambda x: 100 * len(x) / len(df) )
    
    ax[1].set(ylabel="(%)")
    ax[1].set(xlabel="Loan Issued Month")
#-----------------------------------------------------------------------------------------


def plotLoanStat2(df, colors):
    """

    """
    
    df1 = df[['grade','int_rate','loan_status']].sort_values('grade')
    df2 = df[['sub_grade','int_rate','loan_status']].sort_values('sub_grade')

    fig = plt.figure(figsize=(14,10))

    p1 = plt.subplot(211, title = '')
    sns.barplot('grade', 'int_rate',data = df1,
                hue="loan_status", palette=colors)
    plt.ylabel("Interest Rate %")
    plt.xlabel("Grade")

    p2 = plt.subplot(212, title = '')
    sns.barplot('sub_grade', 'int_rate',
                data = df2,
                hue='loan_status', palette=colors)
    plt.ylabel("Interest Rate %")
    plt.xlabel("Sub Grade")

#-------------------------------------------------------------------------------------

def plotLoanStat3(df):
    """

    """
    fig = plt.figure(figsize=(15,5))
    g = sns.boxplot(x='issue_month', y='int_rate', hue='home_ownership',
                    data=df, palette="Set2")
    g.set_xticklabels(g.get_xticklabels(),rotation=45)
    g.set_xlabel("Home Ownership Type", fontsize=12)
    g.set_ylabel("Interest Rate", fontsize=12)
    g.set_title("Interest Rate Distribution vs Month by Home Ownership", fontsize=14)
    plt.legend(loc='upper left')
    

#-----------------------------------------------------------------------------------


def LoanStats2Plotly(df):
    """
    Didn't work through function call! :(
    It is too much bulky!
    """
    
    ave_good_loan_by_purpose = df[df.loan_status == 'Good Loan'].groupby('purpose').loan_amnt.mean().astype(int)
    ave_bad_loan_by_purpose = df[df.loan_status == 'Bad Loan'].groupby('purpose').loan_amnt.mean().astype(int)

    good_bars = go.Bar(
        x = list(ave_good_loan_by_purpose.index),
        y = list(ave_good_loan_by_purpose.values),
        name='Good Loans',
        text='%',
        marker=dict(
            color='rgba(50, 171, 96, 0.7)',
            line = dict(
                color='rgba(50, 171, 96, 1.0)',
                width=2
            )
        )
    )

    bad_bars = go.Bar(
        x = list(ave_bad_loan_by_purpose.index),
        y = list(ave_bad_loan_by_purpose.values),
        name = 'Bad Loans', text='%',
        marker=dict(
            color='rgba(219,64,82,0.7)',
            line = dict(
                color='rgba(219, 64, 82, 1.0)',
                width=2
            )
        )
    )

    data = [good_bars, bad_bars]

    layout = go.Layout(
        title='Average Amount of Loan given for Different Purposes Classified by Loan Status',
        xaxis=dict(
            title=''
        ),
        yaxis=dict(
            title='Average Loan Amount',
        ),
        paper_bgcolor='rgba(250,200,200,0.3)',
        plot_bgcolor='rgba(250,200,200,0.3)',
        showlegend=True
    )

    fig = dict(data=data, layout=layout)
    iplot(fig)
#--------------------------------------------------------------------------------------
    
    
def CPlot(corr_mat,
          axis_labs,
          cmap = "PuBu",
          pad = 0.05,
          rad = 250,
          xlab = 'x',
          ylab = 'y',
          fs = 12,
          xtick_lab_rot = 75,
          ytick_lab_rot = 0):

    """
        This function generates correlation plot given correlation matrix (n, n).
        ------------------------
        
        Input:  corr_mat    - correlation matrix, np.ndarray of dim (n,n)
                axis_labels - custom labels for axis, list of length n
                cmap        - color map
                pad         - for separation of cbar from plot
                rad         - determines sircle radii
                
        Output: figure        
         
    """
    
    n = len(axis_labs)
    #init_plotting() #invokes global plot parameters 
    xtick_pos = []
    ytick_pos = []
    c = []
    
    #separate matrix into 3 sep. arrays
    for i in range(n):
        x = i + 0.5   
        for j in range(n):
            y = j + 0.5  
            xtick_pos.append(x)
            ytick_pos.append(y)    
            c.append(corr_mat[i,j])
    
    #set tick labels
    plt.xticks(xtick_pos[0:n], axis_labs,
               rotation = xtick_lab_rot, fontsize = fs)

    plt.yticks(ytick_pos[0:n], axis_labs,
               rotation = ytick_lab_rot ,fontsize = fs)       
    
    #set axis labels
    plt.xlabel(xlab, fontsize = fs)
    plt.ylabel(ylab, fontsize = fs)
    
    #set axis range
    plt.xlim((0, n))
    plt.ylim((0, n))
    
    norm = plt.Normalize(-1,1)
    plt.scatter(xtick_pos, ytick_pos, c = c, s = rad, norm = norm, cmap = cmap)
    cbar = plt.colorbar(pad = 0.06)
    #cbar.outline.set_visible(False)
    #plt.text(n + 0.75 , n + 0.75, r"$C_{ij}$", fontdict = None, fontsize=fs)


def init_plotting():
    plt.rcParams['font.size'] = 14
    #plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.labelsize'] = 1.5*plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.5*plt.rcParams['font.size']
    plt.rcParams['axes.linewidth'] = 1.1
    plt.rcParams['axes.edgecolor'] = 'k'
    plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['xtick.minor.size'] = 2
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.major.size'] = 5
    plt.rcParams['ytick.minor.size'] = 2
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.loc'] = 'upper left'
    plt.rcParams['xtick.top'] = False
    plt.rcParams['ytick.right'] = False
    plt.rcParams['xtick.direction'] ='out'
    plt.rcParams['ytick.direction'] ='out'
    plt.rcParams['font.weight'] ='bold'
