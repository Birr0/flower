import numpy as np
import sympy
import string
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams
import pandas as pd
import csv
from os.path import join as pjoin
from tqdm import tqdm

def split_by_punctuation(s):
    """
    Convert a string into a list, where the string is split by punctuation,
    excluding underscores or full stops.
    
    For example, the string 'he_ll*o.w0%rl^d' becomes
    ['he_ll', '*', 'o.w0', '%', 'rl', '^', 'd']
    
    Args:
        :s (str): The string to split up
        
    Returns
        :split_str (list[str]): The string split by punctuation
    
    """
    pun = string.punctuation.replace('_', '') # allow underscores in variable names
    pun = string.punctuation.replace('.', '') # allow full stops
    pun = pun + ' '
    where_pun = [i for i in range(len(s)) if s[i] in pun]
    if len(where_pun) > 0:
        split_str = [s[:where_pun[0]]]
        for i in range(len(where_pun)-1):
            split_str += [s[where_pun[i]]]
            split_str += [s[where_pun[i]+1:where_pun[i+1]]]
        split_str += [s[where_pun[-1]]]
        if where_pun[-1] != len(s) - 1:
            split_str += [s[where_pun[-1]+1:]]
    else:
        split_str = [s]
    return split_str

def is_float(s):
    """
    Function to determine whether a string has a numeric value
    
    Args:
        :s (str): The string of interest
        
    Returns:
        :bool: True if s has a numeric value, False otherwise
        
    """
    try:
        float(eval(s))
        return True
    except:
        return False

def replace_floats(s):
    """
    Replace the floats in a string by parameters named b0, b1, ...
    where each float (even if they have the same value) is assigned a
    different b.
    
    Args:
        :s (str): The string to consider
        
    Returns:
        :replaced (str): The same string, but with floats replaced by parameter names
        :values (list[float]): The values of the parameters in order [b0, b1, ...]
        
    """
    split_str = split_by_punctuation(s)
    values = []
    for i in range(len(split_str)):
        if is_float(split_str[i]) and "." in split_str[i]:
            values.append(float(split_str[i]))
            split_str[i] = f'b{len(values)-1}'
        elif len(split_str[i]) > 1 and split_str[i][-1] == 'e' and is_float(split_str[i][:-1]):
            if split_str[i+1] in ['+', '-']:
                values.append(float(''.join(split_str[i:i+3])))
                split_str[i] = f'b{len(values)-1}'
                split_str[i+1] = ''
                split_str[i+2] = ''
            else:
                assert split_str[i+1].is_digit()
                values.append(float(''.join(split_str[i:i+2])))
                split_str[i] = f'b{len(values)-1}'
                split_str[i+1] = ''
    replaced = ''.join(split_str)
    return replaced, values


def load_fun_csv(fname, loss_col=None):
    """
    Load the function CSV file, optionally keeping only the best entry per length.
    
    If loss_col is given, only the row with the best value of that column is kept
    for each unique length. Lower is better unless the column name contains 'R2', 
    in which case higher is better. If loss_col is None, all rows are
    returned as-is.
    
    Args:
        :fname (str): Path to the CSV file
        :loss_col (str, default=None): Column name to use when selecting the best
            entry per length. If None, no deduplication is performed.
        
    Returns:
        :df (pd.DataFrame): DataFrame, deduplicated by length if loss_col is given
    """
    df = pd.read_csv(fname, delimiter=';')
    if loss_col is not None:
        higher_is_better = any(m in loss_col for m in ['R2'])
        if higher_is_better:
            idx = df.groupby('Length')[loss_col].idxmax()
        else:
            idx = df.groupby('Length')[loss_col].idxmin()
        df = df.loc[idx].reset_index(drop=True)
    return df


def convert_operon_fun(eq, names):
    """
    Given the function outputted by operon, express this so that
    the variables are now appropriately names and the floats are
    replaced by parameters.
    
    Args:
        :eq (str): The equation outputted by operon
        :names (list[str]): The names of the parameters in order passed to operon
    
    Returns:
        :new_eq (str): The equation with the replaced symbols and floats
        :values (list[float]): The values of the parameters in order [b0, b1, ...]
    
    """
    
    new_eq = split_by_punctuation(eq)
    for i, n in enumerate(names):
        new_eq = [n if b == f'X{i+1}' else b for b in new_eq]
    new_eq = ''.join(new_eq)
    new_eq = sympy.sympify(new_eq)
    new_eq, values = replace_floats(str(new_eq))
    
    return new_eq, values


def plot_pareto(out_dir, names, ilen=None, objective='MSE', loss_col=None):
    """
    Make the Pareto front plot
    
    Args:
        :out_dir (str): The path to the output directory
            then this is taken to be the final equation
        :names (list[str]): The names of the parameters in order passed to operon
        :ilen (int, default=None): The length of the equation to highlight. If None,
            then this is taken to be the final equation
        :objective (str, default='MSE'): The objective to plot.
        :loss_col (str, default=None): Column name to use when selecting the best
            entry per length. If None, no deduplication is performed.
            
    Returns:
        :fig (matplotlib.figure.Figure): Figure containing Pareto front
        :ax (matplotlib.pyplot.axis): Axis of fig containing the Pareto front
    """
    
    fname = f'{out_dir}/fun.csv'
    df = load_fun_csv(fname, loss_col=loss_col)
    
    if ilen is None:
        eq_idx = -1
    else:
        eq_idx = list(df['Length']).index(ilen)
    best_eq = list(df['Equation'])[eq_idx]
    print('\nAll model lengths:')
    print(list(df['Length']))
    print('\nEquation requested:')
    try:
        display(best_eq)
    except Exception as e:
        print(e)
        print(best_eq)
    eq, pars = convert_operon_fun(best_eq, names)
    print('\nConverted equation:')
    try:
        display(sympy.sympify(eq))
    except Exception as e:
        print(e)
        print(eq)
    print('\nNumber of parameters:', len(pars))
    print('Values:', pars)
        
    # Make a nicer visual version
    param_dict = {'sig8':'sigma_8', 'Om':'Omega_m', 'Ob':'Omega_b', 'ns':'n_s', 
                  'ksigma':'k_sigma', 'neff':'n_e', 'mnu':'m_nu', 'w0':'w_0', 'wa':'w_a'}
    for i, n in enumerate(names):
        if n in param_dict.keys():
            names[i] = param_dict[n]
    eq, pars = convert_operon_fun(best_eq, names)
    eq = sympy.sympify(eq)
    print(eq)
    print('\nLatex version:')
    sympy.print_latex(eq)
    
    rcParams['font.size'] = 16
    
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('Set1')
    ax.axvline(df['Length'].to_numpy()[eq_idx], ls=':', color='k', label='Chosen')
    if object == 'MSE':
        m = np.isfinite(np.sqrt(df[f'{objective}_train']))
        ax.plot(df['Length'][m], np.sqrt(df[f'{objective}_train'])[m], marker='.', color=cmap(0), label='Training')
    else:
        m = np.isfinite(df[f'{objective}_train'])
        ax.plot(df['Length'][m], df[f'{objective}_train'][m], marker='.', color=cmap(0), label='Training')
    if object == 'MSE':
        m = np.isfinite(np.sqrt(df[f'{objective}_test']))
        ax.plot(df['Length'][m], np.sqrt(df[f'{objective}_test'])[m], marker='.', color=cmap(1), label='Validation')
    else:
        m = np.isfinite(df[f'{objective}_test'])
        ax.plot(df['Length'][m], df[f'{objective}_test'][m], marker='.', color=cmap(1), label='Validation')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Model Length')

    if objective == 'MSE':
        ax.set_ylabel('Root Mean Squared Error')
    elif objective == 'MAE':
        ax.set_ylabel('Mean Absolute Error')
    elif objective == 'R2':
        ax.set_ylabel('R2')
    elif objective == 'logL':
        ax.set_ylabel('Log Likelihood')
    else:
        ax.set_ylabel(objective)

    if objective != 'logL':
        ax.set_yscale('log')

    fig.align_labels()
    fig.tight_layout()

    return fig, ax


def prediction_plots(out_dir, ilen=None, loss_col=None):
    """
    Show the difference between the truth and predicted
    
    Args:
        :ini_file (str): The path to the output directory
        :ilen (int, default=None): The length of the equation to highlight. If None,
            then this is taken to be the final equation
        :loss_col (str, default=None): Column name to use when selecting the best
            entry per length. If None, no deduplication is performed.
            
    Returns:
        :fig (matplotlib.figure.Figure): Figure containing plot
        :axs (np.ndarray[matplotlib.pyplot.axis]): Axes of fig containing the plot
    """
    
    fname = f'{out_dir}/fun.csv'
    df = load_fun_csv(fname, loss_col=loss_col)
    
    if ilen is None:
        eq_idx = -1
    else:
        eq_idx = list(df['Length']).index(ilen)
    length = list(df['Length'])[eq_idx]
        
    outname_pred_train = f'{out_dir}/train_{length}.csv'
    outname_pred_test = f'{out_dir}/test_{length}.csv'
    
    cmap = plt.get_cmap('Set1')
    ms = 8
    rcParams['font.size'] = 16
        
    fig, axs = plt.subplots(2, 1, figsize=(6,8), sharex=True)
    
    for i, name in enumerate(['train', 'test']):

        fname= f'{out_dir}/{name}_{length}.csv'
        data = np.loadtxt(fname)
        ytrue = data[:,-2]
        ypred = data[:,-1]
        
        label = ['Training', 'Validation']
        label = label[i]
        
        axs[0].plot(ytrue, ypred, '.', ms=ms, label=label, color=cmap(i))
        axs[1].plot(ytrue, ypred - ytrue, '.', ms=ms, label=label, color=cmap(i))
        
        all_res = ypred - ytrue
        rmse = np.sqrt(np.mean((all_res) ** 2))
        print(f"\n{name}")
        print("\tRMSE:", rmse)
        rmae = np.mean(np.abs(all_res))
        print("\tRMAE:", rmae)
            
    axs[1].set_xlabel('True')
    axs[0].set_ylabel('Predicted')
    axs[1].set_ylabel('Absolute error')
    axs[1].axhline(0, color='k')
    axs[0].legend()
    
    xlim = axs[0].get_xlim()
    ylim = axs[0].get_ylim()
    x = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
    axs[0].plot(x, x, color='k')
    axs[0].set_ylim(ylim)
    axs[0].set_xlim(xlim)
    axs[1].set_xlim(xlim)
        
    fig.align_labels()
    fig.tight_layout()
    
    return fig, axs


def print_to_latex(out_dir, names, loss_col=None):
    """
    Convert operon output to latex and print to file
    
    Args:
        :out_dir (str): The path to the output directory
        :names (list[str]): The names of the parameters in order passed to operon
        :loss_col (str, default=None): Column name to use when selecting the best
            entry per length. If None, no deduplication is performed.
    """
    
    fname = f'{out_dir}/fun.csv'
    df = load_fun_csv(fname, loss_col=loss_col)

    fname = f'{out_dir}/latex.txt'
    with open(fname, 'w') as f:
        for i in tqdm(range(len(df))):
            best_eq = list(df['Equation'])[i]
            length = list(df['Length'])[i]
            for i, n in enumerate(names):
                best_eq = best_eq.replace(f'X{i+1}',n)
            best_eq = sympy.sympify(best_eq)
            replaced, values = replace_floats(str(best_eq))
            expr = sympy.sympify(replaced)
            print(length, ' & $', sympy.latex(expr), '$ \\\\', file=f)
    
    return


def error_plots_2d(out_dir, ilen=None, loss_col=None):
    """
    Show the difference between the truth and predicted in 2D plane
    
    Args:
        :ini_file (str): The path to the output directory
        :ilen (int, default=None): The length of the equation to highlight. If None,
            then this is taken to be the final equation
        :loss_col (str, default=None): Column name to use when selecting the best
            entry per length. If None, no deduplication is performed.
            
    Returns:
        :fig (matplotlib.figure.Figure): Figure containing plot
        :axs (np.ndarray[matplotlib.pyplot.axis]): Axes of fig containing the plot
    """

    fname = f'{out_dir}/fun.csv'
    df = load_fun_csv(fname, loss_col=loss_col)
    
    if ilen is None:
        eq_idx = -1
    else:
        eq_idx = list(df['Length']).index(ilen)
    length = list(df['Length'])[eq_idx]
        
    outname_pred_train = f'{out_dir}/train_{length}.csv'
    outname_pred_test = f'{out_dir}/test_{length}.csv'
    
    cmap = plt.get_cmap('Set1')
    ms = 8
    rcParams['font.size'] = 16

    fig, axs = plt.subplots(1, 2, figsize=(10,4), sharex=True, sharey=True)

    vmin = None
    vmax = None

    for i, name in enumerate(['train', 'test']):

        fname= f'{out_dir}/{name}_{length}.csv'
        data = np.loadtxt(fname)
        ytrue = data[:,-2]
        ypred = data[:,-1]

        z = data[:,1]
        Om = data[:,0]
        error = ypred - ytrue

        if vmin is None:
            vmin = np.min(error)
            vmax = np.max(error)
        
        label = ['Training', 'Validation']
        label = label[i]
        
        pc = axs[i].tricontourf(Om, z, error, cmap='coolwarm', vmin=vmin, vmax=vmax)
        cb = fig.colorbar(pc, ax=axs[i])
        cb.set_label('Absolute Error')
        axs[i].set_xlabel(r'$\Omega_{\rm m}$')
        axs[i].set_ylabel(r'$z$')
        axs[i].set_title(name)
        
        all_res = ypred - ytrue
        rmse = np.sqrt(np.mean((all_res) ** 2))
        print(f"\n{name}")
        print("\tRMSE:", rmse)
        rmae = np.mean(np.abs(all_res))
        print("\tRMAE:", rmae)
    
    fig.align_labels()
    fig.tight_layout()

    return fig, axs

# Here we define some useful sympy variables

basis_functions = [["x", "b"],  # type0
                ["square", "exp", "inv", "sqrt", "log", "cos"],  # type1
                ["+", "*", "-", "/", "pow"]]  # type2

x, y = sympy.symbols('x y', positive=True)
a, b = sympy.symbols('a b', real=True)
sympy.init_printing(use_unicode=True)
inv = sympy.Lambda(a, 1/a)
square = sympy.Lambda(a, a*a)
cube = sympy.Lambda(a, a*a*a)
sqrt = sympy.Lambda(a, sympy.sqrt(a))
log = sympy.Lambda(a, sympy.log(a))
power = sympy.Lambda((a,b), sympy.Pow(a, b))

sympy_locs = {"inv": inv,
            "square": square,
            "cube": cube,
            "pow": power,
            "Abs": sympy.Abs,
            "x":x,
            "sqrt":sqrt,
            "log":log,
            }

'''if __name__ == "__main__":
    out_dir = 'output/'
    fig, ax = plot_pareto(out_dir)
    fig.savefig(f'{out_dir}/pareto.pdf')
    fig, axs = prediction_plots(out_dir)
    fig.savefig(f'{out_dir}/prediction.pdf')'''