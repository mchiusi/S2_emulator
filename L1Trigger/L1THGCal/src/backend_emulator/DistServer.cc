// #include "L1Trigger/L1THGCal/interface/backend_emulator/DistServer.h"

// #include <algorithm>
// #include<iostream>

// using namespace std;
// using namespace l1thgcfirmware;

// DistServer::DistServer( unsigned int nInputs, unsigned int nOutputs, unsigned int nInterleaving ) : 
    // nInputs_(nInputs),
    // nOutputs_(nOutputs),
    // nInterleaving_(nInterleaving),
    // inputs_(nInputs_) {
        // for ( unsigned int iInput=0; iInput<this->nInputs(); ++iInput ) {
            // addr_.emplace_back(this->nInterleaving(),0);
            // for ( unsigned int iInterleave=0; iInterleave<this->nInterleaving(); ++iInterleave ) {
                // addr_[iInput][iInterleave] = iInterleave;
            // }
        // }
    // }

// HGCalTriggerCellSAPtrCollection DistServer::clock(HGCalTriggerCellSAPtrCollection& data) {
    // for ( unsigned int iInput=0; iInput<nInputs(); ++iInput ) {
        // if ( data[iInput]->dataValid() ) {
            // inputs()[iInput].push_back( data[iInput] );
        // }
    // }
    
    // vector< vector< bool > > lMap(nInputs(), vector<bool>(nOutputs()) );

    // HGCalTriggerCellSAPtrCollection lInputs(nInputs(),std::make_shared<HGCalTriggerCell>());

    // std::vector< std::vector< unsigned int> >& addr = this->addr();

    // for ( unsigned int iInput = 0; iInput<nInputs(); ++iInput ) {
        // unsigned int lAddr = addr[iInput][0];
        // if ( lAddr < inputs()[iInput].size() ) {
            // lInputs[iInput] = inputs()[iInput][lAddr];
            // lMap[iInput][ lInputs[iInput]->sortKey() ] = true;
        // }
    // }
    
    // for ( unsigned int iInput = 0; iInput<nInputs(); ++iInput ) {
        // vector< unsigned int>& toRotate = addr[iInput];
        // rotate(toRotate.begin(),  toRotate.begin()+1, toRotate.end() );
    // }

    // HGCalTriggerCellSAPtrCollection lOutputs(nOutputs(),std::make_shared<HGCalTriggerCell>());

    // unsigned int nOutputs = 0;
    // for ( unsigned int iOutput = 0; iOutput<lOutputs.size(); ++iOutput ) {
        // for ( unsigned int iInput = 0; iInput<nInputs(); ++iInput ) {
            // if ( lMap[iInput][iOutput] ) {
                // lOutputs[iOutput] = lInputs[iInput];
                // addr[iInput].back() += this->nInterleaving();
                // nOutputs++;
                // break;
            // }
        // }
    // }


    // return lOutputs; 
// }
