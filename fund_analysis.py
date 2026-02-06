import numpy as np
from scipy.stats import skew,kurtosis
import math 
import matplotlib.pyplot as plt

Risk_Free_Rate=2  # 2% annual risk free rate

def load_data(path):
    try:
        return np.loadtxt(path)
    except Exception as e:
        raise RuntimeError(f"Error Loading{path}: {e}")
    
def descriptive_stats(data):
    return{
     "mean":np.mean(data),
     "median":np.median(data),
     "std":np.std(data),
     "variance":np.var(data,ddof=1),
     "min":np.min(data),
     "max":np.max(data),
     "range":np.max(data)-np.min(data),
     "skewness":skew(data),
     "kurtosis":kurtosis(data),
     "standard_error":np.std(data)/math.sqrt(len(data)),   
     "years":len(data)
    }   

def sharpe_ratio(mean_return,std_dev,rf=Risk_Free_Rate):
    return(mean_return-rf)/std_dev

def print_report(name,stats,sharpe):
    print(f"\n====={name}=====")
    for key,value in stats.items():
        print(f"{key.capitalize():15}: {value:.4f}")
    print(f"Sharpe Ratio    :{sharpe:.4f}")    

def plot_distribution(gfunds,vfunds):
    plt.hist(gfunds,bins=10,alpha=0.6,label="Growth Funds")
    plt.hist(vfunds,bins=10,alpha=0.6,label="Value Funds")
    plt.xlabel("Annual Returns(%)")
    plt.ylabel("Frequency")
    plt.title("Return Distribution Comparison")
    plt.legend()
    plt.show()

def main():
    gfunds=load_data("Gfunds.csv")
    vfunds=load_data("Vfunds.csv")
    
    g_stats=descriptive_stats(gfunds)
    v_stats=descriptive_stats(vfunds)

    g_sharpe=sharpe_ratio(g_stats["mean"],g_stats["std"])
    v_sharpe=sharpe_ratio(v_stats["mean"],v_stats["std"])

    print_report("Growth Funds",g_stats,g_sharpe)
    print_report("Value Funds",v_stats,v_sharpe)

    plot_distribution(gfunds,vfunds)


if __name__=="__main__":
    main()    


        
        
        
