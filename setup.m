% Prepares your matlab workspace for using OpenHandwrittenRecognition.
global G_STARTUP;
global MODEL_FOLD_NAME;
global RANK_LIST_FILE;

if isempty(G_STARTUP)
  G_STARTUP = true;

  incl = {'classifier', 'extractor'};
  for i = 1:length(incl)
    addpath(genpath(fullfile(incl{i})));
  end
  
  
  MODEL_FOLD_NAME = 'model';
  if ~exist(MODEL_FOLD_NAME,'dir'), mkdir(MODEL_FOLD_NAME); end
  
  RANK_LIST_FILE = 'ranklist.mat';

  disp('OpenHandwrittenRecognition is ready.');
  
  clear i incl;
end