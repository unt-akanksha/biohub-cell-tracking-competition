"""Fixed shallow boosted presence trees with a portable numerical evaluator."""
import numpy as np

SETTINGS=dict(loss='log_loss',n_estimators=100,learning_rate=.05,max_depth=2,
              min_samples_leaf=20,subsample=1.,random_state=244691,n_iter_no_change=None)


def features(data):
    values=np.column_stack([data['context'],data['offset']])
    if values.ndim!=2 or values.shape[1]!=8 or not np.isfinite(values).all():raise ValueError('Eight finite fixed presence inputs required')
    return values


def predict(model,x):
    # sklearn trees compare float32-converted inputs against double thresholds.
    x=np.asarray(x,dtype=np.float32).astype(np.float64)
    if x.ndim!=2 or x.shape[1]!=8 or not np.isfinite(x).all():raise ValueError('Finite portable tree inputs required')
    eta=np.full(len(x),model['initial_log_odds'],dtype=float)
    for tree in model['trees']:
        indices=np.zeros(len(x),dtype=np.int64)
        left=np.asarray(tree['left']);right=np.asarray(tree['right']);feature=np.asarray(tree['feature']);threshold=np.asarray(tree['threshold']);value=np.asarray(tree['value'])
        for _ in range(model['settings']['max_depth']+1):
            active=np.flatnonzero(left[indices]>=0)
            if not len(active):break
            current=indices[active]
            indices[active]=np.where(x[active,feature[current]]<=threshold[current],left[current],right[current])
        if (left[indices]>=0).any():raise ValueError('Tree depth exceeds declared bound')
        eta+=model['settings']['learning_rate']*value[indices]
    if not np.isfinite(eta).all():raise ValueError('Nonfinite presence logits')
    return eta


def fit(data,role):
    if role!='fitting':raise ValueError('Diagnostic examples must not enter tree fitting')
    import sklearn
    from sklearn.ensemble import GradientBoostingClassifier
    if sklearn.__version__!='1.9.0':raise ValueError('Audited installed sklearn1.9.0 required')
    x=features(data);labels=np.asarray(data['present'])
    if labels.shape!=(len(x),) or set(labels)!={0,1}:raise ValueError('Known-present and known-absent fitting labels required')
    native=GradientBoostingClassifier(**SETTINGS).fit(x,labels)
    prior=native.init_.class_prior_
    model=dict(settings=SETTINGS,sklearn_version=sklearn.__version__,initial_log_odds=float(np.log(prior[1]/prior[0])),
        feature_order=['best_physical','physical_logsumexp','best_joint','joint_margin','conditional_entropy','nearest_feature_cosine','ranking_disagreement_um','original_presence_log_odds'],
        classes=native.classes_.tolist(),fitting_examples=len(x),fitting_present=int(labels.sum()),trees=[])
    for estimator in native.estimators_[:,0]:
        tree=estimator.tree_
        model['trees'].append(dict(left=tree.children_left.tolist(),right=tree.children_right.tolist(),feature=tree.feature.tolist(),
                                  threshold=tree.threshold.tolist(),value=tree.value[:,0,0].tolist()))
    expected=native.decision_function(x);actual=predict(model,x)
    if not np.allclose(expected,actual,rtol=0,atol=1e-12):raise ValueError('Portable fitting-logit replay failed')
    model['fitting_replay_max_abs']=float(np.max(np.abs(expected-actual)));model['native_parameters']=native.get_params()
    return model
