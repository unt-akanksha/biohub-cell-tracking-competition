"""Vectorized event geometry with a per-movie immutable-input context.

The feature schema and arithmetic follow trajectory_event_features_v1. This
implementation must pass exact float32 replay against frozen source features
before use in scoring or submission. No learned coefficient is changed here.
"""
import numpy as np

from research.trajectory_event_features_v1 import FEATURES
from research.trajectory_disagreement_data_v1 import positions,adjacency


class FeatureContext:
    def __init__(self,graph,edge_matrix):
        self.edge=np.asarray(edge_matrix)
        if self.edge.ndim!=2 or self.edge.shape[1]!=18 or not np.isfinite(self.edge).all():
            raise ValueError('Invalid frozen edge feature schema')
        nodes={int(k):v for k,v in graph['nodes'].items()}
        self.ids=sorted(nodes);self.index={ident:i for i,ident in enumerate(self.ids)}
        pos=positions(nodes)
        self.pos=np.array([pos[i] for i in self.ids],dtype=np.float64)
        self.times=np.array([nodes[i]['t'] for i in self.ids])
        incoming,outgoing=adjacency(graph['edges'])
        self.previous=np.full(len(self.ids),-1,np.int64)
        self.following=np.full(len(self.ids),-1,np.int64)
        for i,ident in enumerate(self.ids):
            if len(incoming[ident])==1:
                previous=incoming[ident][0]
                if nodes[ident]['t']-nodes[previous]['t']==1:self.previous[i]=self.index[previous]
            if len(outgoing[ident])==1:
                following=outgoing[ident][0]
                if nodes[following]['t']-nodes[ident]['t']==1:self.following[i]=self.index[following]
        extent=np.array([63*1.625,255*.40625,255*.40625])
        self.boundary=np.minimum(self.pos.min(axis=1),(extent-self.pos).min(axis=1))
        if np.any(self.boundary < -1e-6):raise ValueError('Node lies outside the observed image geometry')
        self.boundary=np.clip(self.boundary,0.,20.)/10.

    def features(self,problem):
        options=np.asarray(problem['options']);ix=np.asarray(problem['edge_indices'])
        if ix.shape!=(len(options),2) or np.any(ix< -1) or np.any(ix>=len(self.edge)):
            raise ValueError('Invalid event edge lookup')
        x=np.zeros((len(options),len(FEATURES)),np.float64)
        for slot in (0,1):
            valid=ix[:,slot]>=0
            x[valid,:18]+=self.edge[ix[valid,slot]]
        parents=np.array([self.index[int(i)] for i in problem['parents']],np.int64)
        children=np.array([self.index[int(i)] for i in problem['children']],np.int64)
        birth=options[:,0]<0
        death=(options[:,0]>=0)&(options[:,1]<0)
        fork=options[:,2]>=0
        continuation=(options[:,0]>=0)&(options[:,1]>=0)&~fork
        x[continuation,18]=1.;x[fork,19]=1.;x[birth,20]=1.;x[death,21]=1.
        x[birth,28]=self.boundary[children[options[birth,1]]]
        x[death,29]=self.boundary[parents[options[death,0]]]
        rows=np.flatnonzero(fork)
        if len(rows):
            mother=parents[options[rows,0]];a=children[options[rows,1]];b=children[options[rows,2]]
            pm,pa,pb=self.pos[mother],self.pos[a],self.pos[b]
            separation=np.linalg.norm(pa-pb,axis=1)
            x[rows,22]=separation/10.
            x[rows,23]=np.abs(np.linalg.norm(pa-pm,axis=1)-np.linalg.norm(pb-pm,axis=1))/10.
            previous=self.previous[mother];known=previous>=0
            velocity=pm[known]-self.pos[previous[known]]
            x[rows[known],24]=np.linalg.norm((pa[known]+pb[known])/2-pm[known]-velocity,axis=1)/10.
            x[rows[known],25]=1.
            after_a,after_b=self.following[a],self.following[b];known=(after_a>=0)&(after_b>=0)
            x[rows[known],26]=(np.linalg.norm(self.pos[after_a[known]]-self.pos[after_b[known]],axis=1)-separation[known])/10.
            x[rows[known],27]=1.
        if not np.isfinite(x).all():raise ValueError('Nonfinite event evidence')
        return x.astype(np.float32)
