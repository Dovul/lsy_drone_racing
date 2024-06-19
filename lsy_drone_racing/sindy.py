from sindy_rl.dynamics import EnsembleSINDyDynamicsModel

dyna_config = {
    'callbacks': 'project_cartpole',
    'dt': 1,
    'discrete': True,
    
    # Optimizer config 
    'optimizer': {
      'base_optimizer': {
        'name': 'STLSQ',
        'kwargs': {
          'alpha': 5.0e-5,
          'threshold': 7.0e-3,
            },
      },
      # Ensemble Optimization config
      'ensemble': {
        'bagging': True,
        'library_ensemble': True,
        'n_models': 20,
      },
    },
    # Dictionary/Libary Config
    'feature_library': {
      'name': 'affine', # use affine functions
      'kwargs': {
        'poly_deg': 2,
        'n_state': 5 ,
        'n_control': 1,
        'poly_int': True,
        'tensor': True,
      }
    }
}

dyn_model = EnsembleSINDyDynamicsModel(dyna_config)