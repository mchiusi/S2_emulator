import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.colors import LinearSegmentedColormap
import scipy.stats as stats
from scipy.optimize import curve_fit
import numpy as np
import yaml
import json
import mplhep

white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0,    '#ffffff'),
    (1e-10,'#440053'),
    (0.2,  '#404388'),
    (0.4,  '#2a788e'),
    (0.6,  '#21a784'),
    (0.8,  '#78d151'),
    (1,    '#fde624'),
], N=1000)

with open('config.yaml', "r") as afile:
    cfg = yaml.safe_load(afile)["s2emu_config"]

def define_bin(r_z, phi=0):
    return int((r_z-440)/64), 23+int(124*phi/np.pi)

def bin2coord(r_z_bin, phi_bin):
    return 64*(r_z_bin)+440, np.pi*(phi_bin-23)/124

def calculate_shift(heatmap, ev):
    max_bin = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    max_r_z, max_phi = bin2coord(max_bin[0]+0.5, max_bin[1]+0.5)
    r_over_z = np.tan(2*np.arctan(np.exp(-ev.eta_gen)))
    return [r_over_z-max_r_z*ev.LSB_r_z, 
            ev.phi_gen - max_phi,
            np.sum(heatmap)/ev.pT_gen]

def create_plot(objects, step, ev, args, clusters=[]):
    heatmap, seed = np.zeros((64, 124)), []
    
    for bin in objects:
      if (step=='unpacking') and (bin.energy()>0):
        if args.phi: heatmap[define_bin(bin.rOverZ(), ev.LSB_phi*bin.phi()+ev.offset_phi)] += bin.energy()*ev.LSB
        elif args.col: heatmap[define_bin(bin.rOverZ())[0], bin.index()] += bin.energy()*ev.LSB
      
      elif (bin.S()>0):
        heatmap[bin.sortKey(), bin.index()] += (bin.S())*ev.LSB
        if bin.maximaOffset() == cfg['fanoutWidths'][bin.sortKey()]+4 and step=='seeding': 
          seed.append([bin.sortKey(), bin.index(), bin.S()*ev.LSB, distance([bin.sortKey(), bin.index()],ev)])    

    cl = [[get_eta((cl.wroz_.value_/cl.w_.value_)*ev.LSB_r_z), (cl.wphi_.value_/cl.w_.value_)*ev.LSB_phi+ev.offset_phi, \
           cl.e_.value_*ev.LSB] for cl in clusters]
    if len([i[3] for i in seed if i[3] < 0.05]) == 3:
        print(f'3 seeds found for event {ev.event}, (pT, \u03B7, \u03C6)=({ev.pT_gen:.0f}, {ev.eta_gen:.2f},{ev.phi_gen:.2f})') 
        create_heatmap(heatmap, step, ev, seed)

    step_ = {'unpacking': ("columns_" + step if args.col else step, []),
             'seeding': (step, seed),
             'clustering': (step, [[cl.sortKey_, cl.sortKey2_, cl.e_.value_*ev.LSB] for cl in clusters]) }
    if args.col or args.phi: create_heatmap(heatmap, ev, *step_[step]) 
    if args.performance: return calculate_shift(heatmap, ev)
    if args.thr_seed and seed: return [len(seed), ev.eta_gen, ev.pT_gen]
    if (args.cl_energy or args.simulation) and cl: return {'emul_cl':cl, 'CMSSW_ev':ev}

def hgcal_limits(ev):
    plt.axhline(y=(0.476/ev.LSB_r_z-440)/64, color='red', linestyle='--')
    plt.text(3, ((0.476+0.015)/ev.LSB_r_z-440)/64, 'Layer1', color='red', fontsize=6)
    plt.axhline(y=(0.462/ev.LSB_r_z-440)/64, color='red', linestyle='--')
    plt.text(3, ((0.462-0.02)/ev.LSB_r_z-440)/64, 'Layer27', color='red', fontsize=6)
    plt.axhline(y=(0.085/ev.LSB_r_z-440)/64, color='red', linestyle='--')
    plt.text(3, ((0.085+0.015)/ev.LSB_r_z-440)/64, 'Layer1', color='red', fontsize=6)
    plt.axhline(y=(0.076/ev.LSB_r_z-440)/64, color='red', linestyle='--')
    plt.text(3, ((0.076-0.02)/ev.LSB_r_z-440)/64, 'Layer27', color='red', fontsize=6)

def add_markers(markers, title):
    for marker in markers:
      if title=='seeding': 
        plt.scatter(marker[1], marker[0], color='white', marker='o', s=35)
        plt.text(marker[1], marker[0], str(int(marker[2])), fontsize=6, va='center', ha='center')
      if title=='clustering':
        plt.scatter(marker[1], marker[0], color='green', marker='*', s=25, alpha=0.6)
        plt.text(marker[1], marker[0]+1.5, str(int(marker[2])), fontsize=8, va='center', ha='center')

def create_heatmap(heatmap, gen, title, markers=[]):
    plt.imshow(heatmap, cmap=white_viridis, origin='lower', aspect='auto')
    x_tick_labels = [int(val) for val in np.linspace(-30, 150, num=7)]
    y_tick_labels = ['{:.2f}'.format(val) for val in np.linspace(440*gen.LSB_r_z, (64**2+440)*gen.LSB_r_z, num=8)]
    plt.xticks(np.linspace(0, 123, num=7), labels=x_tick_labels)
    plt.yticks(np.linspace(1, 64,  num=8), labels=y_tick_labels)
    plt.colorbar(label='Transverse Energy [GeV]')
    plt.xlabel('\u03C6 (degrees)')
    plt.ylabel('r/z')
    plt.scatter(23+int(124*gen.phi_gen/np.pi), (np.tan(2*np.arctan(np.exp(-gen.eta_gen)))/gen.LSB_r_z-440)/64, 
                color='red', marker='x', s=50)
    add_markers(markers, title)
    plt.title(f'{title} - Event {gen.event} \n pT:{gen.pT_gen:.0f} GeV, \u03B7:{gen.eta_gen:.2f}, \u03C6:{gen.phi_gen:.2f}'.replace('_', ' '))
    plt.grid()
    hgcal_limits(gen)
    plt.savefig(f'plots/single_events/{gen.event}_{title}.pdf')
    plt.savefig(f'plots/single_events/{gen.event}_{title}.png')
    plt.clf()

def produce_plots(shift):
    create_histo([r_z[0] for r_z in shift], 'r_z', 'unpacking')
    create_histo([phi[1] for phi in shift], 'phi', 'unpacking')
    create_histo([p_t[2] for p_t in shift], 'p_t', 'unpacking')

def create_histo(data, variable, title):
    plt.hist(data,  bins=25, alpha=0.5, label='unpacking')
    plt.legend()
    xlabel = (r'$\phi_{bin}^{max energy} - \phi_{gen particle}$' if variable == 'phi' else
              r'$\frac{r}{z}_{bin}^{max energy} - \frac{r}{z}_{gen particle}$' if variable == 'r_z' else
              r'$p^{T}_{bin} / p^{T}_{gen particle}$' )
    plt.xlabel(xlabel)
    plt.ylabel('Counts')
    plt.title('Histogram '+ title + ' ' + xlabel)
    plt.savefig('plots/histogram_'+ variable + '_' + title +'.pdf')
    plt.savefig('plots/histogram_'+ variable + '_' + title +'.png')
    plt.clf()

#########################################################
################## Efficiency plots #####################
#########################################################

def compute_efficiency_plots(seeds, variable, thr, bin_n=10):
    bin_edges = np.linspace(min(variable), max(variable), num=bin_n+1)
    indices = np.digitize(variable, bin_edges) - 1
 
    eff, lo_err, up_err = {}, {}, {}
    for index in range(bin_n):
      """k is number of successes, n is number of trials"""
      bin_indices = np.where(indices == index)[0]
      seeds_bin = [seeds[i] for i in bin_indices]
      k, n = sum(1 for x in seeds_bin if x >= 1), len(seeds_bin)
      if n == 0: eff[index], lo_err[index], up_err[index] = 0, 0, 0; continue

      result = stats.binomtest(k, n, p=k/n)
      eff[index] = k/n
      lo_err[index] = k/n - result.proportion_ci(confidence_level=0.95).low
      up_err[index] = result.proportion_ci(confidence_level=0.95).high - k/n

    plt.errorbar((bin_edges[1:] + bin_edges[:-1])/2, eff.values(), 
                 yerr=np.array(list(zip(lo_err.values(), up_err.values()))).T,
                 xerr=(bin_edges[1] - bin_edges[0])/2, fmt='o', capsize=3, label=thr, alpha=0.7) 

def produce_efficiency_plots(variable, args):
    plt.legend()
    plt.grid()
    plt.xlabel('identified seeds' if variable=='thr' else r'$p_{T}$ [GeV]' if variable=='pT' else r'$\eta$')
    plt.ylabel('Counts' if variable=='thr' else '% of identified seeds')
    plt.title('Efficiency 2D seeds '+variable+' '+args.particles+' '+args.pileup)
    thresholds = '_a'+'_'.join(map(str, [int(i*10) for i in cfg['thresholdMaximaParam_a']]))
    plt.savefig('plots/'+args.particles+'_'+args.pileup+'_efficiency_'+variable+thresholds+'.pdf')
    plt.savefig('plots/'+args.particles+'_'+args.pileup+'_efficiency_'+variable+thresholds+'.png')
    plt.clf()

def plot_seeds(seeds, args):
    n_params = len(cfg['thresholdMaximaParam_a'])
    seeds_list, p_t_list, eta_list, thr_list = [], [], [], []
    for index, thr in enumerate(cfg['thresholdMaximaParam_a']):
        seeds_list.append([seed[0] for idx, seed in enumerate(seeds) if idx%n_params == index])
        eta_list.append([eta[1] for idx, eta in enumerate(seeds) if idx%n_params == index])
        p_t_list.append([p_t[2] for idx, p_t in enumerate(seeds) if idx%n_params == index])
        thr_list.append('a:'+str(thr)+' GeV')
    
    plt.hist(seeds_list, bins=4, label=thr_list) 
    produce_efficiency_plots('thr', args)

    for thr in range(len(thr_list)):
        compute_efficiency_plots(seeds_list[thr], p_t_list[thr], thr_list[thr], 25)
    produce_efficiency_plots('pT', args)
 
    for thr in range(len(thr_list)):
        compute_efficiency_plots(seeds_list[thr], eta_list[thr], thr_list[thr], 5)
    produce_efficiency_plots('eta', args)

#########################################################
############# Checking Cluster Energy ###################
#########################################################

def distance(bin_, gen, in_bin=True):
    if in_bin: 
      r_z_bin, phi_bin = bin2coord(bin_[0]+0.5, bin_[1]+0.5)
      eta_bin = -np.log(np.tan((r_z_bin*0.7/4096)/2))
    else: 
      eta_bin, phi_bin = bin_[0], bin_[1]
    return np.sqrt((eta_bin-gen.eta_gen)**2+(phi_bin-gen.phi_gen)**2)

def scatter_cluster_energy(cl, args):
    clusters, eta_list, p_t_list = [], [], []
    for cluster in cl:
      if cluster: 
        dist = [distance(cl, cluster['CMSSW_ev']) for cl in [[cl[0], cl[1]] for cl in cluster['emul_cl']]]
        clusters.append(cluster['emul_cl'][dist.index(min(dist))][2])
        eta_list.append(cluster['CMSSW_ev'].eta_gen)
        p_t_list.append(cluster['CMSSW_ev'].pT_gen)
   
    plt.scatter(p_t_list, clusters) #, label=thr_list) 
    plt.title(args.pileup+' '+args.particles)
    plt.grid(linestyle='--')
    plt.xlabel(r'$p^{T}_{gen}$')
    plt.ylabel(r'$p^{T}_{cluster}$')
    plt.savefig('plots/'+args.particles+'_'+args.pileup+'_scatter_pT_vs_cluster.pdf')
    plt.savefig('plots/'+args.particles+'_'+args.pileup+'_scatter_pT_vs_cluster.png')
    plt.clf()

#########################################################
########### Comparison with CMSSW Simulation ############
#########################################################

def gaussian(x, A, mu, s):
    return A * np.exp(-((x-mu)**2)/(2*s**2))

def fit_response(data, bin_width = 0.025):
    bin_edges = np.arange(min(data), max(data) + bin_width, bin_width)
    counts, bin_edges = np.histogram(data, bins=bin_edges, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    popt, pcov = curve_fit(gaussian, bin_centers, counts, [max(counts), np.mean(data), np.std(data)])
    amplitude, mean, std = popt
    return abs(std)/mean, np.sqrt(pcov[2, 2])/mean

def compute_responses(emu, simul, gen, args, var, bin_n=10, range_=[0,200], pt=[]):
    bin_edges = np.linspace(range_[0], range_[1], num=bin_n+1)
    indices = np.digitize(gen, bin_edges) - 1

    resp_emu, err_resp_emu, resol_emu, err_resol_emu = {}, {}, {}, {}
    resp_simul, err_resp_simul, resol_simul, err_resol_simul = {}, {}, {}, {}
    for index in range(bin_n):
      bin_idx = np.where(indices == index)[0]
      resp_bin_emu, resp_bin_simul   = ([emu[i]/gen[i] for i in bin_idx], [simul[i]/gen[i] for i in bin_idx]) if var=='pT' else \
                                       ([emu[i]/pt[i]  for i in bin_idx], [simul[i]/pt[i]  for i in bin_idx]) if var=='pT_eta' else \
                                       ([emu[i] for i in bin_idx], [simul[i] for i in bin_idx]) if var=='n_cl_pt' or var=='n_cl_eta' else \
                                       ([emu[i]-gen[i] for i in bin_idx], [simul[i]-gen[i] for i in bin_idx])

      resp_emu[index]       = np.mean(resp_bin_emu) if len(resp_bin_emu)>0 else 0
      err_resp_emu[index]   = np.std(resp_bin_emu)/np.sqrt(len(resp_bin_emu)) if len(resp_bin_emu)>0 else 0
      resp_simul[index]     = np.mean(resp_bin_simul) if len(resp_bin_simul)>0 else 0
      err_resp_simul[index] = np.std(resp_bin_simul)/np.sqrt(len(resp_bin_simul)) if len(resp_bin_simul)>0 else 0

      resol_emu[index]       = np.std(resp_bin_emu)/np.mean(resp_bin_emu) if len(resp_bin_emu)>1 else 0
      err_resol_emu[index]   = np.std(resp_bin_emu)/(np.sqrt(2*len(resp_bin_emu)-2)*np.mean(resp_bin_emu)) if len(resp_bin_emu)>1 else 0
      resol_simul[index]     = np.std(resp_bin_simul)/np.mean(resp_bin_simul) if len(resp_bin_simul)>1 else 0
      err_resol_simul[index] = np.std(resp_bin_simul)/(np.sqrt(2*len(resp_bin_emu)-2)*np.mean(resp_bin_simul)) if len(resp_bin_simul)>1 else 0
    
      if args.fit_resp and (var == 'pT' or var == 'pT_eta'): 
        resol_emu[index], err_resol_emu[index] = fit_response(resp_bin_emu)
        resol_simul[index], err_resol_simul[index] = fit_response(resp_bin_simul)
        # if var == 'pT': plot_bin_distribution(resp_bin_emu, resp_bin_simul, var, index, args)

    # plotting
    plt.style.use(mplhep.style.CMS)
    plt.errorbar((bin_edges[1:] + bin_edges[:-1])/2, resp_emu.values(), 
                 yerr=np.array(list(zip(err_resp_emu.values(), err_resp_emu.values()))).T,
                 xerr=(bin_edges[1] - bin_edges[0])/2, ls='None', lw=2, marker='s', label='emulation') 
    plt.errorbar((bin_edges[1:] + bin_edges[:-1])/2, resp_simul.values(), 
                 yerr=np.array(list(zip(err_resp_simul.values(), err_resp_simul.values()))).T,
                 xerr=(bin_edges[1] - bin_edges[0])/2, ls='None', lw=2, marker='s', label='simulation') 
    plt.ylabel(r'$\phi^{cluster}-\phi^{gen}$' if var=='phi' else r'$\eta^{cluster}-\eta^{gen}$' if var=='eta' else \
               r'$<cluster>$' if var=='n_cl_pt' or var=='n_cl_eta' else r'$p_{T}^{cluster}/p_{T}^{gen}$')
    plt.xlabel(r'$p_{T}^{gen}$ [GeV]' if var=='pT' or var=='n_cl_pt' else r'$\phi^{gen}$' if var=='phi' else r'$|\eta^{gen}|$')
    mplhep.cms.label('Preliminary', data=True, rlabel=args.pileup+' '+args.particles)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('plots/'+args.particles+'_'+args.pileup+'_response_'+var+'.pdf')
    plt.savefig('plots/'+args.particles+'_'+args.pileup+'_response_'+var+'.png')
    plt.clf()

    if var=='n_cl_pt' or var=='n_cl_eta' or var=='eta' or var=='phi': return
    plt.errorbar((bin_edges[1:] + bin_edges[:-1])/2, resol_emu.values(), 
                 yerr=np.array(list(zip(err_resol_emu.values(), err_resol_emu.values()))).T,
                 xerr=(bin_edges[1] - bin_edges[0])/2, ls='None', lw=2, marker='s', label='emulation') 
    plt.errorbar((bin_edges[1:] + bin_edges[:-1])/2, resol_simul.values(), 
                 yerr=np.array(list(zip(err_resol_simul.values(), err_resol_simul.values()))).T,
                 xerr=(bin_edges[1] - bin_edges[0])/2, ls='None', lw=2, marker='s', label='simulation') 
    plt.ylabel(r'$\sigma^{cluster}/\mu^{cluster}$')
    plt.xlabel(r'$p_{T}^{gen}$ [GeV]' if var=='pT' else r'$\phi^{gen}$' if var=='phi' else r'$|\eta^{gen}|$')
    mplhep.cms.label('Preliminary', data=True, rlabel=args.pileup+' '+args.particles)
    plt.legend()
    plt.grid()
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig('plots/'+args.particles+'_'+args.pileup+'_resolution_'+var+'.pdf')
    plt.savefig('plots/'+args.particles+'_'+args.pileup+'_resolution_'+var+'.png')
    plt.clf()

def plot_bin_distribution(resp_emu, resp_simul, var, index, args):
    plt.style.use(mplhep.style.CMS)
    plt.hist(resp_emu, bins=10, alpha=0.5, label='emulation')
    plt.hist(resp_simul, bins=10, alpha=0.5, label='simulation')
    plt.xlabel(r'$p_{T}^{cluster}/p_{T}^{gen}$') 
    plt.title(str(index)+' bin_number')
    plt.legend()
    plt.grid()

    plt.savefig('plots/'+args.particles+'_'+args.pileup+'_'+var+'_distribution_histo_bin'+str(index)+'.pdf')
    plt.clf()

def get_eta(r_z):
    return -np.log(np.tan(np.arctan(r_z)/2)) 

def comparison_histo(emu, simul, args, var, bin_n, range_):
    plt.style.use(mplhep.style.CMS)
    bin_edges = np.linspace(range_[0], range_[1], num=bin_n+1)
    plt.hist(emu,   bins=bin_edges, alpha=.8, label='emulation')
    plt.hist(simul, bins=bin_edges, alpha=.8, label='simulation')
    plt.legend()
    plt.xlabel(r'$p_{T}^{cluster}/p_{T}^{gen}$' if var=='scale_pT' else r'$\phi^{cluster}-\phi^{gen}$' if var=='scale_phi' else \
               r'$\eta^{cluster}-\eta^{gen}$' if var=='scale_eta' else r'$p_{T}^{cluster}$ [GeV]' if var=='pT' else \
               r'$\phi^{cluster}$' if var=='phi' else r'$|\eta^{cluster}|$')
    plt.ylabel('Counts')
    mplhep.cms.label('Preliminary', data=True, rlabel=args.pileup+' '+args.particles)
    if (var=='pT' or var=='eta' or var=='phi') and args.pileup=='PU200': plt.yscale('log')
    plt.savefig('plots/'+args.particles+'_'+args.pileup+'_'+var+'_distribution_histo.pdf')
    plt.savefig('plots/'+args.particles+'_'+args.pileup+'_'+var+'_distribution_histo.png')
    plt.clf()

def plot_simul_comparison(clusters, args):
    n_cl_emu, n_cl_CMSSW, p_t_emu, p_t_CMSSW, p_t_glob_emu, p_t_glob_CMSSW = [], [], [], [], [], []
    eta_emu, phi_emu, eta_CMSSW, phi_CMSSW, eta_glob_emu, eta_glob_CMSSW, phi_glob_emu, phi_glob_CMSSW = [], [], [], [], [], [], [], []
    p_t_gen, eta_gen, phi_gen = [], [], []
    for cl_ev in clusters:
      if cl_ev:
        n_cl_emu.append(len(cl_ev['emul_cl']))
        n_cl_CMSSW.append(len(cl_ev['CMSSW_ev'].cluster.good_cl3d_pt))
        
        # emulation
        cl_pt   = [cl[2] for cl in cl_ev['emul_cl']]
        indices = [i for i, cl in enumerate(cl_pt) if distance([cl_ev['emul_cl'][i][0], \
                   cl_ev['emul_cl'][i][1]],cl_ev['CMSSW_ev'],0)<0.1]
        if len(indices) == 0: continue
        index_  = indices[[cl_pt[i] for i in indices].index(max([cl_pt[i] for i in indices]))]

        if (cl_ev['emul_cl'][index_][0]/cl_ev['CMSSW_ev'].phi_gen < 0.8): print(cl_ev['CMSSW_ev'].event)
        eta_emu.append(cl_ev['emul_cl'][index_][0])
        phi_emu.append(cl_ev['emul_cl'][index_][1])
        p_t_emu.append(cl_ev['emul_cl'][index_][2])

        eta_glob_emu.extend([cl[0] for cl in cl_ev['emul_cl']]) 
        phi_glob_emu.extend([cl[1] for cl in cl_ev['emul_cl']]) 
        p_t_glob_emu.extend([cl[2] for cl in cl_ev['emul_cl']])

        # simulation
        cl_pt   = [cl for cl in cl_ev['CMSSW_ev'].cluster.good_cl3d_pt]
        indices = [i for i, cl in enumerate(cl_pt) if distance([cl_ev['CMSSW_ev'].cluster.good_cl3d_eta[i], \
                   cl_ev['CMSSW_ev'].cluster.good_cl3d_phi[i]],cl_ev['CMSSW_ev'],0)<0.1]
        if len(indices) == 0: continue
        index_  = indices[[cl_pt[i] for i in indices].index(max([cl_pt[i] for i in indices]))]

        p_t_CMSSW.append(1.*cl_ev['CMSSW_ev'].cluster.good_cl3d_pt[index_])
        eta_CMSSW.append(1.*cl_ev['CMSSW_ev'].cluster.good_cl3d_eta[index_])
        phi_CMSSW.append(1.*cl_ev['CMSSW_ev'].cluster.good_cl3d_phi[index_])

        eta_glob_CMSSW.extend([1.*cl for cl in cl_ev['CMSSW_ev'].cluster.good_cl3d_eta])
        phi_glob_CMSSW.extend([1.*cl for cl in cl_ev['CMSSW_ev'].cluster.good_cl3d_phi])
        p_t_glob_CMSSW.extend([1.*cl for cl in cl_ev['CMSSW_ev'].cluster.good_cl3d_pt])

        p_t_gen.append(1.*cl_ev['CMSSW_ev'].pT_gen)
        eta_gen.append(1.*cl_ev['CMSSW_ev'].eta_gen)
        phi_gen.append(1.*cl_ev['CMSSW_ev'].phi_gen)

    # store data before plotting
    plotting_dict = {
        'p_t_emu'     : p_t_emu,      'p_t_CMSSW'     : p_t_CMSSW, 
        'p_t_glob_emu': p_t_glob_emu, 'p_t_glob_CMSSW': p_t_glob_CMSSW,
        'eta_emu'     : eta_emu,      'phi_emu'       : phi_emu, 
        'eta_CMSSW'   : eta_CMSSW,    'phi_CMSSW'     : phi_CMSSW,
        'eta_glob_emu': eta_glob_emu, 'eta_glob_CMSSW': eta_glob_CMSSW, 
        'phi_glob_emu': phi_glob_emu, 'phi_glob_CMSSW': phi_glob_CMSSW,
        'p_t_gen'     : p_t_gen,      'eta_gen'   : eta_gen, 'phi_gen': phi_gen,
        'n_cl_emu'    : n_cl_emu,     'n_cl_CMSSW': n_cl_CMSSW
    }

    file_path = 'plots/data/clusters_data_'+args.particles+'_'+args.pileup+'.json'
    with open(file_path, 'w') as f:
      json.dump(plotting_dict, f)
      print('Json file created in /plots/data')

    plotting_json(args)

def plotting_json(args):
    file_path = 'plots/data/clusters_data_'+args.particles+'_'+args.pileup+'.json'
    with open(file_path, 'r') as f:
      plotting_dict = json.load(f)
      print('Json file read in /plots/data')

    # Access the data
    p_t_emu, p_t_CMSSW = plotting_dict['p_t_emu'], plotting_dict['p_t_CMSSW']
    p_t_glob_emu, p_t_glob_CMSSW = plotting_dict['p_t_glob_emu'], plotting_dict['p_t_glob_CMSSW']
    eta_emu, phi_emu = plotting_dict['eta_emu'], plotting_dict['phi_emu']
    eta_CMSSW, phi_CMSSW = plotting_dict['eta_CMSSW'], plotting_dict['phi_CMSSW']
    eta_glob_emu, eta_glob_CMSSW = plotting_dict['eta_glob_emu'], plotting_dict['eta_glob_CMSSW']
    phi_glob_emu, phi_glob_CMSSW = plotting_dict['phi_glob_emu'], plotting_dict['phi_glob_CMSSW']
    p_t_gen, eta_gen, phi_gen = plotting_dict['p_t_gen'], plotting_dict['eta_gen'], plotting_dict['phi_gen']
    n_cl_emu, n_cl_CMSSW = plotting_dict['n_cl_emu'], plotting_dict['n_cl_CMSSW'] 

    # distributions
    comparison_histo(p_t_glob_emu, p_t_glob_CMSSW, args, 'pT',  20, [0, 200 if args.pileup=='PU0' else 100])
    comparison_histo(eta_glob_emu, eta_glob_CMSSW, args, 'eta', 20, [1.6, 2.8])
    comparison_histo(phi_glob_emu, phi_glob_CMSSW, args, 'phi', 20, [0, 2.2])

    scale_emu, scale_simul = np.divide(p_t_emu, p_t_gen), np.divide(p_t_CMSSW, p_t_gen)
    scale_emu_eta, scale_simul_eta = np.subtract(eta_emu, eta_gen), np.subtract(eta_CMSSW, eta_gen)
    scale_emu_phi, scale_simul_phi = np.subtract(phi_emu, phi_gen), np.subtract(phi_CMSSW, phi_gen)
    comparison_histo(scale_emu, scale_simul, args, 'scale_pT', 30, [0, 1.6])
    comparison_histo(scale_emu_eta, scale_simul_eta, args, 'scale_eta', 20, [-0.01, 0.01])
    comparison_histo(scale_emu_phi, scale_simul_phi, args, 'scale_phi', 20, [-0.02, 0.02])

    # responses - scale and reoslution
    compute_responses(p_t_emu, p_t_CMSSW, p_t_gen, args, 'pT', 10, [0, 200 if args.pileup=='PU0' else 100])
    compute_responses(p_t_emu, p_t_CMSSW, eta_gen, args, 'pT_eta', 10, [1.6,2.8], p_t_gen)
    compute_responses(eta_emu, eta_CMSSW, eta_gen, args, 'eta', 10, [1.6,2.8])
    compute_responses(phi_emu, phi_CMSSW, phi_gen, args, 'phi', 10, [0.2,1.8])
    compute_responses(n_cl_emu, n_cl_CMSSW, p_t_gen, args, 'n_cl_pt',  10, [0, 200 if args.pileup=='PU0' else 100])
    compute_responses(n_cl_emu, n_cl_CMSSW, eta_gen, args, 'n_cl_eta', 10, [1.6,2.8])

    # cluster shift wrt gen particle
    histo_2D_position(scale_emu_eta,   scale_emu_phi,   'emulation',  args)
    histo_2D_position(scale_simul_eta, scale_simul_phi, 'simulation', args)

def histo_2D_position(x_data, y_data, var, args, bins=(20,20), cmap=white_viridis):
    plt.style.use(mplhep.style.CMS)
    plt.hist2d(x_data, y_data, bins=bins, cmap=cmap)
    plt.colorbar()
    mplhep.cms.label('Preliminary', data=True, rlabel=args.pileup+' '+args.particles)
    plt.ylabel(r'$\phi^{emulation}-\phi^{gen}$'  if var=='emulation' else r'$\phi^{emulation}-\phi^{gen}$')
    plt.xlabel(r'$\eta^{simulation}-\eta^{gen}$' if var=='emulation' else r'$\eta^{simulation}-\eta^{gen}$')
    plt.savefig('plots/'+args.particles+'_'+args.pileup+'_histogram2D_eta_phi_'+ var + '.pdf')
    plt.savefig('plots/'+args.particles+'_'+args.pileup+'_histogram2D_eta_phi_'+ var + '.png')
    plt.clf()

## not used ##
def create_plot_py(objects, ev, args):
    heatmap = np.zeros((64, 124))

    for bin in objects: # min col = -23 from S2.ChannelAllocation.xml
        if args.phi: heatmap[define_bin(bin['rOverZ'], ev.LSB_phi*bin['phi']+ev.offset_phi)] += bin['energy']*ev.LSB
        elif args.col: heatmap[define_bin(bin['rOverZ'])[0], 23+bin['column']] += bin['energy']*ev.LSB

    if args.performance: return calculate_shift(heatmap, ev) 
    elif args.col or args.phi: create_heatmap(heatmap, 'columns_pre_unpacking' if args.col else 'pre_unpacking', ev)

