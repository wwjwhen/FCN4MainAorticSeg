filesm = dir(fullfile('img', '*mask.png'));
V = zeros(300, 300, 1);
for i = 525:1400%length(filesm)
    im = imread(['img\', filesm(i).name]);
    im = imcrop(im, [100, 140, 299, 299]);
    im = bwareaopen(im, 50);
    V(:, :, i - 524) = im;
    imshow(im, []);
    pause(0.01);
    fprintf('%d\n', i);
end