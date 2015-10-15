
pattern = cell(10,1);
for idx = 1:10
    pattern{idx} = reshape(Model.F(:,idx),meshSize);
end
implot(pattern{:});