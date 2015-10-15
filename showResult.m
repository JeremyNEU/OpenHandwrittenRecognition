
for idx = 1:25
    testImg = img(:,:,idx);
    label = recogDigits(testImg);
    subplot(5,5,idx); imshow(testImg);
    title(num2str(label));
end