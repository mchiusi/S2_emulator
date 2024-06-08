#include "L1Trigger/L1THGCal/interface/backend_emulator/HGCalHistoClustering_SA.h"
#include "L1Trigger/L1THGCal/interface/backend_emulator/ClusterizerColumn.h"

#include <cmath>
#include <algorithm>
#include <memory>
#include <unordered_map>

using namespace std;
using namespace l1thgcfirmware;

HGCalHistoClustering::HGCalHistoClustering(const ClusterAlgoConfig& config) : config_(config) {}

void HGCalHistoClustering::runClustering(HGCalTriggerCellSAPtrCollection& triggerCells,
                                         HGCalHistogramCellSAPtrCollection& histogram,
                                         HGCalClusterSAPtrCollection& clustersOut ) const {

  HGCalTriggerCellSAPtrCollection triggerCellRamOut;
  HGCalTriggerCellSAPtrCollection clusteredTriggerCellsOut;
  HGCalHistogramCellSAPtrCollection maximaFifoOut;
  clusterizer(triggerCells, histogram, triggerCellRamOut, maximaFifoOut);
  triggerCellToCluster(triggerCells, histogram, clustersOut);
  clusterAccumulator(clustersOut, histogram);
  clusterTree(clustersOut);
}

double Dist( const HGCalTriggerCellSAPtr& TC , const HGCalHistogramCellSAPtr& Maxima, const ClusterAlgoConfig& Config )
{ 
  if ( ( TC == nullptr ) or ( Maxima == nullptr ) ) return UINT_MAX;
  
  // -------------------------------------------------
  // Cartesian for comparison
  double tc_phi = TC->phi_ * M_PI/1944;
  double tc_x = TC->rOverZ_ * std::cos( tc_phi );
  double tc_y = TC->rOverZ_ * std::sin( tc_phi );
  
  double hc_phi = Maxima->X_ * M_PI/1944; // (2.0*M_PI/3.0) / 4096;
  double hc_x = Maxima->Y_ * std::cos( hc_phi );
  double hc_y = Maxima->Y_ * std::sin( hc_phi );
       
  double dx = tc_x - hc_x;
  double dy = tc_y - hc_y;
           
  double dr2 = ( dx * dx ) + ( dy * dy );
  // -------------------------------------------------
                                                   
  // -------------------------------------------------
  // unsigned int r1 = TC->rOverZ_;
  // unsigned int r2 = Maxima->Y_;
  // int dR = r1 - r2;
  // int dPhi = TC->phi_ - Maxima->X_;
  // unsigned int dR2 = dR * dR;
  // const int maxbin = Config.nBinsCosLUT()-1;
  // unsigned int cosTerm = ( abs(dPhi) > maxbin ) ? Config.cosLUT( maxbin ) : Config.cosLUT( abs(dPhi) ); // stored in 10 bit
  // int correction = ( ( ( r1 * r2 ) >> 1 ) * cosTerm ) >> 17;
  // dR2 += correction;
              
  // std::cout << ( dR2 - dr2 ) << " : " << correction << std::endl;
  // -------------------------------------------------
  
  return dr2;
  }

// double Dist( const HGCalTriggerCellSAPtr& TC , const HGCalHistogramCellSAPtr& Maxima, const ClusterAlgoConfig& Config )
// { 
//   if ( ( TC == nullptr ) or ( Maxima == nullptr ) ) return UINT_MAX;
//   
//   unsigned int r1 = TC->rOverZ_;
//   unsigned int r2 = Maxima->Y_;
//   int dR = r1 - r2;
//   int dPhi = TC->phi_ - Maxima->X_;
//   unsigned int dR2 = dR * dR;
//   const int maxbin = Config.nBinsCosLUT()-1;
//   unsigned int cosTerm = ( abs(dPhi) > maxbin ) ? Config.cosLUT( maxbin ) : Config.cosLUT( abs(dPhi) ); // stored in 10 bit
//   int correction = ( ( ( r1 * r2 ) >> 1 ) * cosTerm ) >> 17;
//   dR2 += correction;
//         
//   return dR2;
// }

void HGCalHistoClustering::clusterizer( HGCalTriggerCellSAPtrCollection& triggerCells, HGCalHistogramCellSAPtrCollection& histogram, HGCalTriggerCellSAPtrCollection& triggerCellsRamOut, HGCalHistogramCellSAPtrCollection& maximaFifoOut ) const
{
  std::array< ClusterizerColumn , 124 > lColumns;

  // std::cout << "Inital number of TCs " << triggerCells.size() << std::endl;
  // std::sort(triggerCells.begin(), triggerCells.end(), [](const HGCalTriggerCellSAPtr& a, const HGCalTriggerCellSAPtr& b) {
  //       return a->clock() < b->clock();
  // });

  std::unordered_map<int, int> TC_counter;

  // Map the TCs into the RAM using the LUT
  auto start = triggerCells.front()->clock();
  for( auto& tc : triggerCells ){
    auto frame = tc->clock() - start;
    TC_counter[tc->index()] += 1;
    frame = TC_counter[tc->index()]%216;
    auto& lut_out = config_.TriggerCellAddressLUT( ( 216*tc->index() ) + frame );
    lColumns.at( tc->index() ).MappedTCs.at( lut_out ) = std::move(tc);
    // if ( tc != nullptr and tc->index() == 34 ) { std::cout << "frame " << frame << std::endl; }
  }
  triggerCells.clear();
  
  // Map the maxima into the FIFO
  for( auto& i : histogram ){
    if( i->maximaOffset_>0 or i->left_ or i->right_ ) {
     // std::cout << i->Y_ << " " << i->index_ << std::endl; 
     lColumns.at( i->index_ ).MaximaFifo.push_back( i );
    } 
  }
  histogram.clear();

  // Move the first two entries out of the FIFO into "current" and "next"
  // for( auto& lColumn : lColumns ) lColumn.pop().pop();

  HGCalTriggerCellSAPtrCollection triggerCellsOut;
  HGCalHistogramCellSAPtrCollection histogramOut;

  //Read the tcs out sequentially
  for ( unsigned int frame = 0; frame != 216; ++frame ) {  
    for ( unsigned int iColumn = 0; iColumn != config_.cColumns(); ++iColumn ) {      
      auto& col = lColumns.at( iColumn );
      auto& tc = col.MappedTCs.at( frame );
  
      // Get the maxima from the FIFO  //  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< STILL TRYING TO EMULATE HOW THE FIRMWARE DOES THIS!
      if( col.counter < 0 )
      {
        auto MaximaInFifo = ( ( col.MaximaFifo.size() ) and ( col.MaximaFifo.front()->clock_-306 < frame ) );
        auto NoCurrent     = ( col.Current == nullptr );
        auto TcPastMidpoint = ( tc != nullptr ) and ( tc->rOverZ() > col.midr ); // and ( col.MaximaFifo.front()->clock_-303 < frame ) ) );
        if ( MaximaInFifo and ( NoCurrent or TcPastMidpoint ) ) col.counter = 3;
      }
      else 
      {  
        if( col.counter == 0 ) col.pop();
        if( col.counter >= 0 ) col.counter--;
      }

      // if ( iColumn == 34 ) { std::cout << "col Counter " << col.counter << std::endl;}
      // if ( col.Current != nullptr )
      // {
      //   // std::cout << col.Current << std::endl; 
      //   maximaFifoOut.push_back( std::make_unique< HGCalHistogramCell >( *col.Current ) );
      //   auto& hcx = maximaFifoOut.back();
      //   hcx->clock_ = frame + 289 + 16;
      // }
    
      // Compare the TC against the maxima
      // if ( iColumn == 34 and tc == nullptr ) { std::cout << "nullpointer " << std::endl; }
      if( tc == nullptr ) continue;
     
      // triggerCellsRamOut.push_back( std::make_unique< HGCalTriggerCell >( *tc ) );
      // auto& tcx = triggerCellsRamOut.back();
      // tcx->setClock( frame + 289 + 20 );
      // tcx->setLastFrame( false ); //(frame==215);

      unsigned int CurrentdR2Cut(80000); // Magic numbers
      double CurrentDist = Dist( tc , col.Current, config_ );
      // if ( iColumn == 34 and tc != nullptr ) { std::cout << "Frame " << frame << " r/z bin " << int((tc->rOverZ() - config_.rOverZHistOffset()) / config_.rOverZBinSize()) << " Index "<< tc->index_ << " energy " << tc->energy_ << " CurrentDist " << CurrentDist << "maximum r/z " << col.Current->Y_ << std::endl; }
      // std::cout << col.Current << " Column " << iColumn << " Frame " << frame << " CurrentDist " << CurrentDist << std::endl;
      if( CurrentDist < CurrentdR2Cut )
      {
        HGCalHistogramCellSAPtr hc = std::make_unique< HGCalHistogramCell > ( *col.Current );
        hc->clock_ = frame + 289 + 20;    
        histogramOut.emplace_back( hc );        

        // if ( iColumn == 34 and tc != nullptr ) { std::cout << "##########################Frame " << frame << " r/z bin " << int((tc->rOverZ() - config_.rOverZHistOffset()) / config_.rOverZBinSize()) << " energy " << tc->energy_ << std::endl; }
        tc->clock_ = frame + 289 + 20; 
        tc->sortKey_ = hc->sortKey_;        
        triggerCellsOut.push_back( move(tc) );
      }
    }
  }
  triggerCells = move(triggerCellsOut);
  // std::cout << "Clusterised TCs " << triggerCells.size() << std::endl;
  histogram = move(histogramOut);
}

void HGCalHistoClustering::triggerCellToCluster(const HGCalTriggerCellSAPtrCollection& clusteredTriggerCells,
                                                const HGCalHistogramCellSAPtrCollection& histogram,
                                                HGCalClusterSAPtrCollection& clustersOut) const {

    std::map< std::pair< int , int > , HGCalHistogramCellSAPtr > maxima_map;
    for( auto& x : histogram ) maxima_map[ std::make_pair( x->index_ , x->clock_ ) ] = x;
    

    const unsigned int stepLatency = config_.getStepLatency( TriggerCellToCluster );

    for ( const auto& tc : clusteredTriggerCells ) {

        auto cluster = make_unique<HGCalCluster>(
                            tc->clock() + stepLatency,
                            tc->index(),
                            tc->lastFrame(),
                            tc->dataValid()
        );

        auto lIt1( maxima_map.find( std::make_pair( tc->index() , tc->clock() ) ) );
        auto lIt2( maxima_map.find( std::make_pair( tc->index() , tc->clock()+1 ) ) );    
        cluster->L_ = lIt1->second->left_;
        cluster->R_ = lIt1->second->right_;
        cluster->X_ = ( lIt2 == maxima_map.end() ) or ( lIt1->second->sortKey_ != lIt2->second->sortKey_ );
        
        cluster->sortKey_ = lIt1->second->sortKey_;
        cluster->sortKey2_ = lIt1->second->sortKey2_;


        // Cluster from single TC
        // Does this ever happen?
        // if ( tc->deltaR2_ >= 25000 ) { // Magic numbers
        // clusters.push_back( cluster );
        // continue;
        // }

        uint64_t s_TC_W = ( int( tc->energy() / 4 ) == 0 ) ? 1 : tc->energy() / 4;
        uint64_t s_TC_Z = config_.depth( tc->layer() );

        unsigned int triggerLayer = config_.triggerLayer( tc->layer() );
        
        unsigned int s_E_EM = ( (  ( (uint64_t) tc->energy() * config_.layerWeight_E_EM( tc->layer() ) ) + config_.correction() ) >> 18 );
        if ( s_E_EM > config_.saturation() ) s_E_EM = config_.saturation();

        unsigned int s_E_EM_core = ( ( (uint64_t) tc->energy() * config_.layerWeight_E_EM_core( tc->layer() ) + config_.correction() ) >> 18 );
        if ( s_E_EM_core > config_.saturation() ) s_E_EM_core = config_.saturation();

        // Alternative constructor perhaps?
        cluster->set_n_tc(1);
        cluster->set_n_tc_w(1);

        // std::cout << tc->layer() << " " << triggerLayer << std::endl;
        cluster->set_e((config_.layerWeight_E( tc->layer() ) == 1) ? tc->energy() : 0);
        cluster->set_e_h_early((config_.layerWeight_E_H_early( tc->layer() ) == 1) ? tc->energy() : 0);

        cluster->set_e_em(s_E_EM);
        cluster->set_e_em_core(s_E_EM_core);

        cluster->set_w(s_TC_W);
        cluster->set_w2(s_TC_W * s_TC_W);

        cluster->set_wz(s_TC_W * s_TC_Z);
        cluster->set_wphi(s_TC_W * tc->phi());
        cluster->set_wroz(s_TC_W * tc->rOverZ());

        cluster->set_wz2(s_TC_W * s_TC_Z * s_TC_Z);
        cluster->set_wphi2(s_TC_W * tc->phi() * tc->phi());
        cluster->set_wroz2(s_TC_W * tc->rOverZ() * tc->rOverZ());

        const unsigned nTriggerLayers = 34;  // Should get from config/elsewhere in CMSSW
        cluster->set_layerbits( ( ( (uint64_t) 1) << ( nTriggerLayers - triggerLayer ) ) & 0x3FFFFFFFF);
        cluster->set_sat_tc(cluster->e() == config_.saturation() || cluster->e_em() == config_.saturation());

        cluster->set_shapeq(1);

        clustersOut.push_back( move(cluster) );
    }
}


void HGCalHistoClustering::clusterAccumulator( HGCalClusterSAPtrCollection& clusters, const HGCalHistogramCellSAPtrCollection& histogram ) const
{  
  HGCalClusterSAShrPtrCollection output;
  
  std::map< std::pair< int , int > , HGCalClusterSAShrPtr > cluster_map;

  // std::cout << "###################" << std::endl;
  for( auto& x : clusters ){
    auto lKey = std::make_pair( x->sortKey_ , x->index_ );
    auto lIt = cluster_map.find( lKey );
    if ( lIt == cluster_map.end() ){
      HGCalClusterSAShrPtr lVal = make_shared< HGCalCluster >( *x );
      lVal->X_ = true; // Last entry should always have X_ set
      output.push_back( lVal );
      cluster_map[lKey] = lVal;
    } else {
      // std::cout << "R/Z " << x->sortKey_ << " Index " << x->index_ << " energy " << x->e_.value_ << std::endl;
      *lIt->second += *x;
      lIt->second->L_ = x->L_;
      lIt->second->R_ = x->R_;
      lIt->second->X_ = x->X_;
      lIt->second->sortKey_ = x->sortKey_;
      lIt->second->sortKey2_ = x->sortKey2_;
    }
    // if (x->sortKey_==11) {std::cout << "Index " << x->index_ << " energy " << x->e_.value_ << std::endl;}
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
      // std::cout << sharedPtr->sortKey_ << " final clusters " << sharedPtr->e_.value_ << std::endl;
      clusters.push_back(std::make_unique<HGCalCluster>(*sharedPtr));
  }
}
