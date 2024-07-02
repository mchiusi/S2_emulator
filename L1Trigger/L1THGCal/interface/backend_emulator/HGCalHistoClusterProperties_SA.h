#ifndef __L1Trigger_L1THGCal_HGCalHistoClusterProperties_h__
#define __L1Trigger_L1THGCal_HGCalHistoClusterProperties_h__

#include "L1Trigger/L1THGCal/interface/backend_emulator/HGCalTriggerCell_SA.h"
#include "L1Trigger/L1THGCal/interface/backend_emulator/CentroidHelper.h"
#include "L1Trigger/L1THGCal/interface/backend_emulator/HGCalHistoClusteringConfig_SA.h"
#include "L1Trigger/L1THGCal/interface/backend_emulator/HGCalCluster_SA.h"
// #include "DataFormats/L1THGCal/interface/HGCalCluster_HW.h"

namespace l1thgcfirmware {

  class HGCalHistoClusterProperties {
  public:
    HGCalHistoClusterProperties(const l1thgcfirmware::ClusterAlgoConfig& config);
    ~HGCalHistoClusterProperties() {}

    void runClusterProperties(const l1thgcfirmware::HGCalClusterSAPtrCollection& protoClustersIn,
                              const l1thgcfirmware::CentroidHelperPtrCollection& readoutFlags,
                              HGCalClusterSAPtrCollection& clustersOut) const;

  private:
    // Cluster property steps
    void clusterSum(const l1thgcfirmware::HGCalClusterSAPtrCollection& protoClusters,
                    const l1thgcfirmware::CentroidHelperPtrCollection& readoutFlags,
                    l1thgcfirmware::HGCalClusterSAPtrCollection& clusterAccumulation,
                    l1thgcfirmware::HGCalClusterSAPtrCollection& clusterSums) const;
    // void clusterProperties(l1thgcfirmware::HGCalClusterSAPtrCollection& clusterSums) const;

    // Helper functions
    // unsigned int sigma_coordinate(unsigned int Sum_W,
    //                                                        unsigned long int Sum_Wc2,
    //                                                        unsigned int Sum_Wc,
    //                                                        double scale = 0) const;

    // std::vector<int> showerLengthProperties(unsigned long int layerBits) const;

    // double convertRozToEta( HGCalClusterSAPtr& cluster ) const;
    // double convertSigmaRozRozToSigmaEtaEta( HGCalClusterSAPtr& cluster ) const;

    const l1thgcfirmware::ClusterAlgoConfig& config_;
  };
}  // namespace l1thgcfirmware

#endif