#include "L1Trigger/L1THGCal/interface/backend_emulator/HGCalHistoClustering_SA.h"

#include <cmath>
#include <algorithm>

using namespace std;
using namespace l1thgcfirmware;

HGCalHistoClustering::HGCalHistoClustering(const ClusterAlgoConfig& config) : config_(config) {}

void HGCalHistoClustering::runClustering(const HGCalTriggerCellSAPtrCollection& triggerCellsIn,
                                         const HGCalHistogramCellSAPtrCollection& histogramIn,
                                         HGCalTriggerCellSAShrPtrCollection& clusteredTriggerCellsOut,
                                         CentroidHelperPtrCollection& readoutFlagsOut,
                                         HGCalClusterSAPtrCollection& protoClusters) const {
  HGCalTriggerCellSAShrPtrCollection unclusteredTriggerCells;
  clusterizer(triggerCellsIn, histogramIn, clusteredTriggerCellsOut, unclusteredTriggerCells, readoutFlagsOut);
  triggerCellToCluster(clusteredTriggerCellsOut, protoClusters);
  // clusterAccumulator(protoClusters, histogramIn);
  // clusterTree(protoClusters);
}

// Main implementation of clustering
// Takes histogram containing seeds and streams of TCs (each stream corresponds to one column of the histogram)
// Outputs clustered and unclustered TCs, and readoutFlags which contain info on which clustered TCs belong to each cluster
// Require more comments on firmware to provide more meanginful comments here
void HGCalHistoClustering::clusterizer(const HGCalTriggerCellSAPtrCollection& triggerCellsIn,
                                       const HGCalHistogramCellSAPtrCollection& histogram,
                                       HGCalTriggerCellSAShrPtrCollection& clusteredTriggerCellsOut,
                                       HGCalTriggerCellSAShrPtrCollection& unclusteredTriggerCellsOut,
                                       CentroidHelperPtrCollection& readoutFlagsOut) const {
  unsigned int seedCounter = 0;
  CentroidHelperPtrCollections fifos(config_.nFifos());
  vector<unsigned int> clock(config_.cColumns(), config_.clusterizerMagicTime());
  const unsigned dummy_entries_latched = 1;
  CentroidHelperShrPtrCollection latched(
      config_.nFifos() + dummy_entries_latched,
      make_shared<CentroidHelper>());  // 1 extra (dummy) entry compared to fifos, to match firmware behaviour (avoids issues with index wrap-around)

  HGCalTriggerCellSAShrPtrCollections clusteredTriggerCells(config_.cColumns(), HGCalTriggerCellSAShrPtrCollection());
  HGCalTriggerCellSAShrPtrCollections unclusteredTriggerCells(config_.cColumns(), HGCalTriggerCellSAShrPtrCollection());
  CentroidHelperPtrCollections readoutFlags(config_.cColumns());

  HGCalTriggerCellSAShrPtrCollectionss triggerCellBuffers(
      config_.cColumns(), HGCalTriggerCellSAShrPtrCollections(config_.cRows(), HGCalTriggerCellSAShrPtrCollection()));
  int energy_ = 0;
  for (const auto& tc : triggerCellsIn) {
    // Temp copy of tc whilst moving from shared to unique ptr
    // std::cout << tc->index() << " " << tc->sortKey() << std::endl;
    energy_ += tc->energy();
    triggerCellBuffers.at(tc->index()).at(tc->sortKey()).push_back(make_shared<HGCalTriggerCell>(*tc));
  }
  std::cout  << "Global energy " << energy_ << std::endl;

  for (unsigned int iRow = 0; iRow < config_.cRows(); ++iRow) {
    for (unsigned int j = 0; j < config_.nColumnsPerFifo(); ++j) {
      for (unsigned int k = 0; k < config_.nFifos(); ++k) {
        unsigned int col = config_.firstSeedBin() + (config_.nColumnsPerFifo() * k) + j;
        const auto& cell = histogram.at(config_.cColumns() * iRow + col);
        if (cell->S() > 0 and cell->maximaOffset()==(config_.fanoutWidths(cell->sortKey())+3)) {
          // std::cout << cell->index() << " " << cell->sortKey() << std::endl;
          auto ch = make_unique<CentroidHelper>(cell->clock() + 1 + j,
                                                config_.nColumnsPerFifo() * k + j,
                                                cell->index(),
                                                cell->sortKey(),
                                                cell->S(),
                                                cell->X(),
                                                cell->Y(),
                                                true);
          fifos[k].push_back(move(ch));
          ++seedCounter;
        }
      }
    }
  }
  
  std::cout << seedCounter << " seeds" << std::endl;
  while (seedCounter > 0) {
    for (unsigned int i = 0; i < config_.nFifos(); ++i) {
      if (!latched[i]->dataValid()) {
        if (!fifos[i].empty()) {
          latched[i] = move(fifos[i][0]);
          fifos[i].erase(fifos.at(i).begin());
        }
      }
    }

    const unsigned dummy_entries_accepted = 2;
    CentroidHelperShrPtrCollection accepted(
        config_.nFifos() + dummy_entries_accepted,
        make_shared<
            CentroidHelper>());  // 2 extra (dummy) entry compared to fifos (1 compared to latched), to match firmware behaviour (avoids issues with index wrap-around)
    CentroidHelperShrPtrCollection lastLatched(latched);

    for (unsigned int i = 0; i < config_.nFifos(); ++i) {
      // Different implementation to python emulator
      // For i=0, i-1=-1, which would give the last element of lastLatched in python, but is out of bounds in C++
      // Similar for i=17 (==config_.nFifos()-1)
      // Need to find out intended behaviour
      bool deltaMinus =
          (i > 0) ? (lastLatched[i]->column() - lastLatched[i - 1]->column()) > config_.nColumnFifoVeto() : true;
      bool deltaPlus = (i < config_.nFifos() - 1)
                           ? (lastLatched[i + 1]->column() - lastLatched[i]->column()) > config_.nColumnFifoVeto()
                           : true;

      bool compareEMinus = (i > 0) ? (lastLatched[i]->energy() > lastLatched[i - 1]->energy()) : true;
      bool compareEPlus =
          (i < config_.nFifos() - 1) ? (lastLatched[i]->energy() >= lastLatched[i + 1]->energy()) : true;

      if (lastLatched[i]->dataValid()) {
        // Similar out of bounds issue here
        // if ( ( !lastLatched[i+1]->dataValid() || compareEPlus || deltaPlus ) && ( !lastLatched[i-1]->dataValid() || compareEMinus || deltaMinus ) ) {

        bool accept = true;
        if (lastLatched.size() > i + 1) {
          if (!lastLatched[i + 1]->dataValid() || compareEPlus || deltaPlus) {
            accept = true;
          } else {
            accept = false;
          }
        }

        if (i > 0) {
          if (!lastLatched[i - 1]->dataValid() || compareEMinus || deltaMinus) {
            accept = true;
          } else {
            accept = false;
          }
        }

        if (accept) {
          accepted[i] = latched[i];
          latched[i] = make_shared<CentroidHelper>();
          --seedCounter;
        }
      }
    }

    for (const auto& a : accepted) {
      if (a->dataValid()) {
        for (unsigned int iCol = a->column() - config_.nColumnsForClustering();
             iCol < a->column() + config_.nColumnsForClustering() + 1;
             ++iCol) {
          clock[iCol] = clock[a->column()];
        }
      }
    }

    vector<unsigned int> readoutFlagClocks;

    for (const auto& a : accepted) {
      if (a->dataValid()) {
        unsigned int T = 0;
        // clusterizer 
        int clu_energy = 0;
        std::cout << "Considering seed " << a->column() << std::endl;
        for (int iCol = a->column() - config_.nColumnsForClustering();
             iCol < a->column() + config_.nColumnsForClustering() + 1;
             ++iCol) {
          const unsigned stepLatency = 8;
          clock[iCol] += stepLatency;
          for (int k = -1 * config_.nRowsForClustering(); k < int(config_.nRowsForClustering()) + 1; ++k) {
            int row = a->row() + k;
            if (row < 0)
              continue;
            if (row >= int(config_.cRows()))
              continue;  // Not in python emulator, but required to avoid out of bounds access
            if (triggerCellBuffers[iCol][row].empty()) {
              // std::cout << "Opsss " << iCol << " row " << row << std::endl;
              clock[iCol] += 1;
              continue;
            }
            for (auto& tc : triggerCellBuffers[iCol][row]) {
              clock[iCol] += 1;
              double tc_phi = tc->phi_ * M_PI/1944;
              double tc_x = tc->rOverZ_ * std::cos( tc_phi );
              double tc_y = tc->rOverZ_ * std::sin( tc_phi );
  
              double hc_phi = a->X() * M_PI/1944; // (2.0*M_PI/3.0) / 4096;
              double hc_x = a->Y() * std::cos( hc_phi );
              double hc_y = a->Y() * std::sin( hc_phi );
       
              double dx = tc_x - hc_x;
              double dy = tc_y - hc_y;
              
              double dR2 = ( dx * dx ) + ( dy * dy );
              // unsigned int r1 = tc->rOverZ();
              // unsigned int r2 = a->Y();
              // int dR = r1 - r2;
              // unsigned int absDPhi = abs(int(tc->phi()) - int(a->X()));
              // unsigned int dR2 = dR * dR;
              // unsigned int cosTerm = (absDPhi > config_.nBinsCosLUT()) ? 2047 : config_.cosLUT(absDPhi);

              // const unsigned a = 128;   // 2^7
              // const unsigned b = 1024;  // 2^10
              // dR2 += int(r1 * r2 / a) * cosTerm / b;
              tc->setClock(clock[iCol] + 1);
              if (clock[iCol] > T)
                T = clock[iCol];

              unsigned int dR2Cut = 5000; // config_.getDeltaR2Threshold(tc->layer());
              if (dR2 < dR2Cut) {
                clusteredTriggerCells[iCol].push_back(tc);
                clu_energy += tc->energy();
              } else {
                unclusteredTriggerCells[iCol].push_back(tc);
              }
            }
          }

          for (const auto& tc : clusteredTriggerCells[iCol]) {
            auto tcMatch = std::find_if(
                triggerCellBuffers[iCol][tc->sortKey()].begin(),
                triggerCellBuffers[iCol][tc->sortKey()].end(),
                [&](const HGCalTriggerCellSAShrPtr tcToMatch) {
                  bool isMatch = tc->index() == tcToMatch->index() && tc->rOverZ() == tcToMatch->rOverZ() &&
                                 tc->layer() == tcToMatch->layer() && tc->energy() == tcToMatch->energy() &&
                                 tc->phi() == tcToMatch->phi() && tc->sortKey() == tcToMatch->sortKey() &&
                                 tc->deltaR2() == tcToMatch->deltaR2() && tc->dX() == tcToMatch->dX() &&
                                 tc->Y() == tcToMatch->Y() && tc->lastFrame() == tcToMatch->lastFrame() &&
                                 tc->dataValid() == tcToMatch->dataValid() && tc->clock() == tcToMatch->clock();
                  return isMatch;
                });

            if (tcMatch != triggerCellBuffers[iCol][tc->sortKey()].end()) {
              triggerCellBuffers[iCol][tc->sortKey()].erase(tcMatch);
            }
          }
        }
        std::cout << "Clusterized energy " << clu_energy << std::endl;

        unsigned int readoutFlagClock = 0;
        for (unsigned int iCol = a->column() - config_.nColumnsForClustering();
             iCol < a->column() + config_.nColumnsForClustering() + 1;
             ++iCol) {
          clock[iCol] = T + 1;

          CentroidHelperPtr readoutFlag = make_unique<CentroidHelper>(T - 2, iCol, true);
          while ( std::find(readoutFlagClocks.begin(), readoutFlagClocks.end(), readoutFlag->clock()) != readoutFlagClocks.end() ) {
            readoutFlag->setClock(readoutFlag->clock() + 1);
          }
          readoutFlagClock = readoutFlag->clock();

          const unsigned stepLatency = 14;
          if (readoutFlag->clock() ==
              config_.clusterizerMagicTime() + stepLatency) {  // Magic numbers - latency of which particular step?
            readoutFlag->setClock(readoutFlag->clock() + 1);
          }

          readoutFlags[iCol].push_back(move(readoutFlag));
        }
        readoutFlagClocks.push_back(readoutFlagClock);
      }
    }
  }

  const unsigned largeReadoutTime = 1000;
  int cl_energy = 0;
  for (unsigned int i = 0; i < largeReadoutTime;
       ++i) {  // Magic numbers - a large number to ensure we read out all clustered trigger cells etc.?
    for (unsigned int iCol = 0; iCol < config_.cColumns(); ++iCol) {
      for (const auto& clustered : clusteredTriggerCells[iCol]) {
        // std::cout << clustered->clock() << "----" << config_.clusterizerMagicTime() + i << std::endl;
        if (clustered->clock() == config_.clusterizerMagicTime() + i) {
          cl_energy += clustered->energy();
          // std::cout<< "filling.." << std::endl;
          clusteredTriggerCellsOut.push_back(clustered);
        }
      }

      for (const auto& unclustered : unclusteredTriggerCells[iCol]) {
        if (unclustered->clock() == config_.clusterizerMagicTime() + i) {
          unclusteredTriggerCellsOut.push_back(unclustered);
        }
      }

      for (auto& readoutFlag : readoutFlags[iCol]) {
        if (readoutFlag) {
          if (readoutFlag->clock() == config_.clusterizerMagicTime() + i) {
            // TODO : Check if we can move the readoutFlag and leave a nullptr
            // Or if the readoutFlag could be used again later on
            readoutFlagsOut.push_back(move(readoutFlag));
          }
        }
      }
    }
  }
  std::cout << "Clusterized energy all seeds " << cl_energy << std::endl;
}

// Converts clustered TCs into cluster object (one for each TC) ready for accumulation
void HGCalHistoClustering::triggerCellToCluster(const HGCalTriggerCellSAShrPtrCollection& clusteredTriggerCells,
                                                HGCalClusterSAPtrCollection& clustersOut) const {
  const unsigned int stepLatency = config_.getStepLatency(TriggerCellToCluster);

  clustersOut.clear();
  for (const auto& tc : clusteredTriggerCells) {
    auto cluster = make_unique<HGCalCluster>(tc->clock() + stepLatency, tc->index(), true, true);

    // Cluster from single TC
    // Does this ever happen?
    // Removed from newer versions of firmware in any case, but leave for now
    const unsigned singleTCDeltaR2Thresh = 25000;
    if (tc->deltaR2() >= singleTCDeltaR2Thresh) {
      clustersOut.push_back(move(cluster));
      continue;
    }

    const unsigned weightFactor = 4;
    unsigned long int s_TC_W = (int(tc->energy() / weightFactor) == 0) ? 1 : tc->energy() / weightFactor;
    unsigned long int s_TC_Z = config_.depth(tc->layer());

    unsigned int triggerLayer = config_.triggerLayer(tc->layer());
    const unsigned nBitsESums = 18;  // Need to double check this is correct description of constant
    unsigned int s_E_EM =
        ((((unsigned long int)tc->energy() * config_.layerWeight_E_EM(triggerLayer)) + config_.correction()) >>
         nBitsESums);
    if (s_E_EM > config_.saturation())
      s_E_EM = config_.saturation();

    unsigned int s_E_EM_core =
        (((unsigned long int)tc->energy() * config_.layerWeight_E_EM_core(triggerLayer) + config_.correction()) >>
         nBitsESums);
    if (s_E_EM_core > config_.saturation())
      s_E_EM_core = config_.saturation();

    // Alternative constructor perhaps?
    cluster->set_n_tc(1);
    cluster->set_n_tc_w(1);

    cluster->set_e((config_.layerWeight_E( tc->layer() ) == 1) ? tc->energy() : 0);
    cluster->set_e_h_early((config_.layerWeight_E_H_early(triggerLayer) == 1) ? tc->energy() : 0);

    cluster->set_e_em(s_E_EM);
    cluster->set_e_em_core(s_E_EM_core);

    cluster->set_w(s_TC_W);
    cluster->set_w2(s_TC_W * s_TC_W);

    cluster->set_wz(s_TC_W * s_TC_Z);
    // cluster->set_weta(0);
    cluster->set_wphi(s_TC_W * tc->phi());
    cluster->set_wroz(s_TC_W * tc->rOverZ());

    cluster->set_wz2(s_TC_W * s_TC_Z * s_TC_Z);
    // cluster->set_weta2(0);
    cluster->set_wphi2(s_TC_W * tc->phi() * tc->phi());
    cluster->set_wroz2(s_TC_W * tc->rOverZ() * tc->rOverZ());

    const unsigned nTriggerLayers = 34;  // Should get from config/elsewhere in CMSSW
    cluster->set_layerbits(cluster->layerbits() | (((unsigned long int)1) << (nTriggerLayers - triggerLayer)));
    cluster->set_sat_tc(cluster->e() == config_.saturation() || cluster->e_em() == config_.saturation());
    cluster->set_shapeq(1);

    // Temp copy of TC whilst reducing use of shared ptr
    cluster->add_constituent(make_shared<HGCalTriggerCell>(*tc));
    clustersOut.push_back(move(cluster));
  }
}

void HGCalHistoClustering::clusterAccumulator( HGCalClusterSAPtrCollection& clusters, const HGCalHistogramCellSAPtrCollection& histogram ) const
{  
  HGCalClusterSAShrPtrCollection output;
  
  std::map< std::pair< int , int > , HGCalClusterSAShrPtr > cluster_map;
  for( auto& x : clusters ){
    std::cout << x->sortKey_ << std::endl;
    auto lKey = std::make_pair( x->sortKey_ , x->index_ );
    auto lIt = cluster_map.find( lKey );
    if ( lIt == cluster_map.end() ){
      HGCalClusterSAShrPtr lVal = make_shared< HGCalCluster >( *x );
      lVal->X_ = true; // Last entry should always have X_ set
      output.push_back( lVal );
      cluster_map[lKey] = lVal;
    } else {
      *lIt->second += *x;
      lIt->second->L_ = x->L_;
      lIt->second->R_ = x->R_;
      lIt->second->X_ = x->X_;
      lIt->second->sortKey_ = x->sortKey_;
      lIt->second->sortKey2_ = x->sortKey2_;
    }
    
  }

  for( auto& x : histogram ){  
    auto lIt = cluster_map.find( std::make_pair( x->sortKey_ , x->index_ ) );
    if ( lIt != cluster_map.end() ) lIt->second->clock_ = x->clock_ + 11;     
  }

  for( auto& x : output ) x->saturate();
 
  std::sort( output.begin() , output.end() , []( const HGCalClusterSAShrPtr& a , const HGCalClusterSAShrPtr& b ){ return std::make_pair( a->clock_ , a->index_ ) < std::make_pair( b->clock_ , b->index_ ); } );

  clusters.clear();
  clusters.reserve(output.size());
  for (auto& sharedPtr : output) {
      clusters.push_back(std::make_unique<HGCalCluster>(*sharedPtr));
  }
}

void HGCalHistoClustering::clusterTree( HGCalClusterSAPtrCollection& clusters ) const
{
  HGCalClusterSAShrPtrCollection output;

  // vvvvvvvvvvvvvvvvvv HACK TO VERIFY VALUES
  std::map< std::pair< int , int > , HGCalClusterSAShrPtr > cluster_map;

  for( auto& x : clusters ){
    auto lKey = std::make_pair( x->sortKey_ , x->sortKey2_ );
    auto lIt = cluster_map.find( lKey );
    if ( lIt == cluster_map.end() ){
      auto lVal = make_shared< HGCalCluster >( *x );
      lVal->index_ = 0;
      lVal->X_ = 0;
      output.push_back( lVal );
      cluster_map[lKey] = lVal;
    } else {
      *lIt->second += *x;
      lIt->second->clock_ = max( lIt->second->clock_ , x->clock_ );      
      lIt->second->L_ |= x->L_;
      lIt->second->R_ |= x->R_;
      // lIt->second->X_ = x->X_;
      lIt->second->sortKey_ = x->sortKey_;
      lIt->second->sortKey2_ = x->sortKey2_;
    }
    // if (x->sortKey_==11) {std::cout << "Index " << x->index_ << " energy " << x->e_.value_ << std::endl;}
  }
  // ^^^^^^^^^^^^^^^^^ HACK TO VERIFY VALUES
  
  for( auto& x : output ){
    x->saturate();
    x->clock_ += 9;
  }

  std::sort( output.begin() , output.end() , []( const HGCalClusterSAShrPtr& a , const HGCalClusterSAShrPtr& b ){ return std::make_pair( a->clock_ , a->index_ ) < std::make_pair( b->clock_ , b->index_ ); } );

  HGCalClusterSAShrPtr last = nullptr;
  for( auto& x : output ){
    if( last != nullptr and x->clock_ <= last->clock_ ) x->clock_ = last->clock_ + 1;
    last = x;
  }

  std::sort( output.begin() , output.end() , []( const HGCalClusterSAShrPtr& a , const HGCalClusterSAShrPtr& b ){ return std::make_pair( a->clock_ , a->index_ ) < std::make_pair( b->clock_ , b->index_ ); } );

  clusters.clear();
  clusters.reserve(output.size());
  for (auto& sharedPtr : output) {
      std::cout << sharedPtr->sortKey_ << " final clusters " << sharedPtr->e_.value_ << std::endl;
      clusters.push_back(std::make_unique<HGCalCluster>(*sharedPtr));
  }
}