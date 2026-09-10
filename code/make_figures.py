"""Draw manuscript figures from the exact archived results used in the tables."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from crossfit import ROOT

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42})
COLORS = ['#85929E','#BC6543','#187D82']


def oracle_plot():
    fig,axes = plt.subplots(1,2,figsize=(9.3,3.7),sharey=True,layout='constrained')
    summary=json.loads((ROOT/'results/oracle.json').read_text())
    for ax,sc,title in zip(axes,('ccmar','seqmar'),('Common validity','Sequential MAR only')):
        vals={k:[] for k in ('full','cc','seq')}
        with (ROOT/'results/replicates'/f'oracle_{sc}.csv').open() as fh:
            for row in csv.DictReader(fh):
                if row['estimator'] in vals:
                    vals[row['estimator']].append(float(row['estimate']))
        box=ax.boxplot(list(vals.values()),tick_labels=['Full data','Coarsened','Sequential'],
                       patch_artist=True,widths=.52,
                       flierprops={'markersize':2,'markeredgecolor':'#697580','alpha':.35},
                       medianprops={'color':'white','linewidth':1.5})
        for patch,color in zip(box['boxes'],COLORS):
            patch.set_facecolor(color)
        ax.axhline(summary[sc]['psi'],color='#343B43',linestyle='--',linewidth=1)
        ax.set_title(title,pad=12)
        ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Estimated average treatment effect')
    path=ROOT/'paper/figures/estimator_comparison.pdf'
    fig.savefig(path)
    fig.savefig(path.with_suffix('.png'),dpi=180)
    plt.close(fig)


def sensitivity_plot():
    data=json.loads((ROOT/'results/sensitivity.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(9.3,3.8),layout='constrained')
    eff=data[:3]
    x=np.array([100*d['population_complete'] for d in eff])
    y=np.array([d['reduction_pct'] for d in eff])
    axes[0].errorbar(x,y,yerr=[y-np.array([d['boot_lo'] for d in eff]),
                              np.array([d['boot_hi'] for d in eff])-y],fmt='o-',color=COLORS[2],capsize=4)
    axes[0].set_xlabel('Population complete records (%)')
    axes[0].set_ylabel('Variance reduction (%)')
    axes[0].set_title('Efficiency under common validity',pad=12)
    ident=data[3:]
    x=[d['gamma'] for d in ident]
    for k,label,color,offset in [('cc','Coarsened',COLORS[1],-.025),('seq','Sequential',COLORS[2],.025)]:
        axes[1].errorbar(np.array(x)+offset,[d[k]['bias'] for d in ident],
                        yerr=[1.96*d[k]['mcse_bias'] for d in ident],fmt='o-',color=color,capsize=4,label=label)
    axes[1].axhline(0,color='#343B43',linestyle='--',linewidth=1)
    axes[1].set_xlabel(r'Coefficient of $L_1$ in response logit ($\gamma$)')
    axes[1].set_ylabel('Bias')
    axes[1].set_title('Identification as CCMAR fails',pad=12)
    axes[1].set_xticks(x)
    axes[1].legend(frameon=False,loc='lower left')
    for ax in axes:
        ax.grid(axis='y',alpha=.15)
    path=ROOT/'paper/figures/sensitivity_summary.pdf'
    fig.savefig(path)
    fig.savefig(path.with_suffix('.png'),dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    (ROOT/'paper/figures').mkdir(parents=True,exist_ok=True)
    oracle_plot()
    sensitivity_plot()
    print('Wrote both figures from archived results.')
