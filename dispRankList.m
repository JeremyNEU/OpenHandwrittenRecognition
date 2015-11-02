global RANK_LIST_FILE;

if exist(RANK_LIST_FILE,'file'), load(RANK_LIST_FILE, '-mat'); end
    
colHeadings = arrayfun(@(x)sprintf('%d',x),0:9,'UniformOutput',false);
format = repmat('%-6s',1,12); % 1 row 12 col
header = sprintf(format,colHeadings{:},'| Overall | Algorithm');
fprintf('\n%s\n%s\n',header,repmat('-',size(header)));

for n = 1:length(ranklist)
    confMat = ranklist{n}.confMat;
    confRateMat = bsxfun(@rdivide,confMat,sum(confMat,2));
    ErrorRate = 1-diag(confRateMat);
    fprintf('%-6.2f', 100*ErrorRate);
    fprintf('|  %-6.2f | ', 100*mean(ErrorRate));
    disp(ranklist{n}.name);
end

% todo: sort overall detection rate and remove reduplicate items.