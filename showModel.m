
pattern = cell(10,1);
for idx = 1:10
    pattern{idx} = reshape(Model.F(:,idx),meshSize);
end
implot(pattern{:});

for idx = 1:10
    subplot(3,4,idx);title(num2str(idx-1));
end

figure;
for idx = 1:10
    subplot(3,4,idx);
    I = mat2gray(pattern{idx});
    BW = im2bw(I, 0.4);
    imshow(BW);
    title(num2str(idx-1));
end