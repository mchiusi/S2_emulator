import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import binomtest
import numpy as np
import yaml

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
        if bin.maximaOffset() == cfg['fanoutWidths'][bin.sortKey()] and step=='seeding': 
          seed.append([bin.sortKey(), bin.index(), bin.S()*ev.LSB, distance([bin.sortKey(), bin.index()],ev)])    

    cl = [[cl.sortKey_, cl.sortKey2_, cl.e_.value_*ev.LSB] for cl in clusters]
    if len([i[3] for i in seed if i[3] < 0.05]) == 3:
        print(f'3 seeds found for event {ev.event}, (pT, \u03B7, \u03C6)=({ev.pT_gen:.0f}, {ev.eta_gen:.2f},{ev.phi_gen:.2f})') 
        create_heatmap(heatmap, step, ev, seed)
  
    step_ = { 'unpacking': ("columns_" + step if args.col else step, []),
              'seeding': (step, seed),
              'clustering': (step, cl) }
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
    plt.grid(linestyle='--')
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

      result = binomtest(k, n, p=k/n)
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

def plot_cluster_energy(cl, args):
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

    compute_ratio_cl(clusters, p_t_list, args, 'p_t', 10)

def compute_ratio_cl(num, den, args, variable='p_t', bin_n=10, binning=None):
    if binning is None: binning = den
    ratio = np.divide(num, den)

    bin_edges = np.linspace(0, max(binning), num=bin_n+1)
    indices = np.digitize(binning, bin_edges) - 1

    result, err, num_, den_ = {}, {}, {}, {}
    for index in range(bin_n):
      bin_idx = np.where(indices == index)[0]
      ratio_bin, num_bin, den_bin = [ratio[i] for i in bin_idx], [num[i] for i in bin_idx], [den[i] for i in bin_idx]
      result[index] = np.mean(ratio_bin) if len(ratio_bin)>0 else 0
      err[index]    = np.std(ratio_bin)/np.sqrt(len(ratio_bin)) if len(ratio_bin)>0 else 0
      num_[index], den_[index] = np.sum(num_bin), np.sum(den_bin)

    thr = int(cfg['thresholdMaximaParam_a'][0])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 2]})
    if variable == 'clusters': 
      ax1.stairs(num_.values(), bin_edges, alpha=0.7, fill=True, label='Emulator' if args.simulation else 'Cluster')
      ax1.stairs(den_.values(), bin_edges, alpha=0.7, fill=True, label='CMSSW' if args.simulation else r'$p_{T}^{gen}$', color='orange')
      ax1.set_ylabel('Number of identified clusters')
    else: 
      ax1.hist(num, bin_edges, alpha=0.7, fill=True, label='Emulator' if args.simulation else 'Cluster')
      ax1.hist(den, bin_edges, alpha=0.7, fill=True, label='CMSSW' if args.simulation else r'$p_{T}^{gen}$', color='orange')
      ax1.set_ylabel(r'$p^{cluster}_{T}$ distribution')
    ax1.grid(linestyle='--')
    ax1.legend()

    ax2.errorbar((bin_edges[1:] + bin_edges[:-1])/2, result.values(), 
                 yerr=np.array(list(zip(err.values(), err.values()))).T, xerr=(bin_edges[1] - bin_edges[0])/2,
                 fmt='o', capsize=3, label=str(thr)+' GeV', alpha=0.7) 
    ax2.set_ylabel(r'$p_{T}^{cluster}/p_{T}^{gen}$' if args.cl_energy else r'$p_{T}^{emulator}/p_{T}^{CMSSW}$' if variable=='p_t' \
                   else r'$n_{cluster}^{emulator}/n_{cluster}^{CMSSW}$')
    ax2.set_xlabel(r'$p_{T}^{gen}$')
    ax2.grid(linestyle='--')
    ax2.legend()
    
    title = args.particles+' '+args.pileup + (' Calibration factors ' if args.cl_energy else ' Emulator vs CMSSW '+variable)
    ax1.set_title(title)
    plt.savefig('plots/'+title.replace(" ", "_")+'_thr'+str(thr)+'.pdf')
    plt.savefig('plots/'+title.replace(" ", "_")+'_thr'+str(thr)+'.png')
    plt.clf()

#########################################################
########### Comparison with CMSSW Simulation ############
#########################################################

def distance_phi(bin_, phi_gen):
    _, phi_bin = bin2coord(0, bin_+0.5)
    return phi_bin-phi_gen

def distance_eta(bin_, eta_gen):
    r_z_bin, _ = bin2coord(bin_+0.5, 0)
    eta_bin = -np.log(np.tan((r_z_bin*0.7/4096)/2))
    return eta_bin-eta_gen

def plot_simul_comparison(clusters, args):
    n_cl_emu, n_cl_CMSSW, p_t_emu, p_t_CMSSW = [], [], [], []
    eta_emu, phi_emu, eta_CMSSW, phi_CMSSW = [], [], [], []
    for cl_ev in clusters:
      if cl_ev:
        n_cl_emu.append(len(cl_ev['emul_cl']))
        n_cl_CMSSW.append(len(cl_ev['CMSSW_ev'].cluster.good_cl3d_pt))
        dist = [distance(cl, cl_ev['CMSSW_ev']) for cl in [[cl[0], cl[1]] for cl in cl_ev['emul_cl']]]
        eta_emu.append(distance_eta(cl_ev['emul_cl'][dist.index(min(dist))][0], cl_ev['CMSSW_ev'].eta_gen))
        phi_emu.append(distance_phi(cl_ev['emul_cl'][dist.index(min(dist))][1], cl_ev['CMSSW_ev'].phi_gen))
        p_t_emu.append(cl_ev['emul_cl'][dist.index(min(dist))][2])

        dist = [distance(cl, cl_ev['CMSSW_ev'], 0) for cl in [[cl_ev['CMSSW_ev'].cluster.good_cl3d_eta[cl], \
                cl_ev['CMSSW_ev'].cluster.good_cl3d_phi[cl]] for cl in range(n_cl_CMSSW[-1])]]
        p_t_CMSSW.append(cl_ev['CMSSW_ev'].cluster.good_cl3d_pt[dist.index(min(dist))])
        eta_CMSSW.append(cl_ev['CMSSW_ev'].cluster.good_cl3d_eta[dist.index(min(dist))]-cl_ev['CMSSW_ev'].eta_gen)
        phi_CMSSW.append(cl_ev['CMSSW_ev'].cluster.good_cl3d_phi[dist.index(min(dist))]-cl_ev['CMSSW_ev'].phi_gen)
    p_t_gen = [cl['CMSSW_ev'].pT_gen for cl in clusters if cl]

    compute_ratio_cl(n_cl_emu, n_cl_CMSSW, args, 'clusters', 20, p_t_gen)
    compute_ratio_cl(p_t_emu, p_t_CMSSW, args, 'p_t', 20, p_t_gen)

    # cluster shift wrt gen particle
    histo_2D_position(eta_emu, phi_emu, 'emu', args)
    histo_2D_position(eta_CMSSW, phi_CMSSW, 'CMSSW', args)
    histo_2D_position(np.subtract(eta_emu, eta_CMSSW), np.subtract(phi_emu, phi_CMSSW), 'comparison', args)

def histo_2D_position(x_data, y_data, title, args, bins=(20,20), cmap=white_viridis):
    plt.hist2d(x_data, y_data, bins=bins, cmap=cmap)
    plt.colorbar()
    plt.title(r'Difference in $\eta$ and $\phi$ emulation vs CMSSW' if title=='comparison' else 'Comparison gen particle position w/'+title)
    plt.xlabel(r'$\eta_{emulator}$' if title=='emu' else r'$\eta_{CMSSW}$' if title=='CMSSW' else r'$\Delta\eta$')
    plt.ylabel(r'$\phi_{emulator}$' if title=='emu' else r'$\phi_{CMSSW}$' if title=='CMSSW' else r'$\Delta\phi$')
    plt.savefig('plots/'+args.particles+'_'+args.pileup+'_histogram2D_eta_phi_'+ title + '.pdf')
    plt.savefig('plots/'+args.particles+'_'+args.pileup+'_histogram2D_eta_phi_'+ title + '.png')
    plt.clf()

## not used ##
def create_plot_py(objects, ev, args):
    heatmap = np.zeros((64, 124))

    for bin in objects: # min col = -23 from S2.ChannelAllocation.xml
        if args.phi: heatmap[define_bin(bin['rOverZ'], ev.LSB_phi*bin['phi']+ev.offset_phi)] += bin['energy']*ev.LSB
        elif args.col: heatmap[define_bin(bin['rOverZ'])[0], 23+bin['column']] += bin['energy']*ev.LSB

    if args.performance: return calculate_shift(heatmap, ev) 
    elif args.col or args.phi: create_heatmap(heatmap, 'columns_pre_unpacking' if args.col else 'pre_unpacking', ev)

