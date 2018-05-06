% filesm = dir(fullfile('img', '*mask.png'));
% fileso = dir(fullfile('img', '*orig.png'));
% %figure(1);
% for i = 3250:3500
%     fprintf('Pic %d\n', i);
%     subplot(1, 2, 1);
%     imgm = imread(['img\' filesm(i).name]);
%     imshow(imgm);
%     subplot(1, 2, 2);
%     imgo = imread(['img\' fileso(i).name]);
%     imshow(imgo);
%     pause(0.01);
% end



samplePaths = dir(fullfile('E:\newdata\', 'XCT*'));
maskPaths = dir(fullfile('E:\newdata\', 'CT*res'));
assert(length(samplePaths) == length(maskPaths), 'sample and mask not aligned');
ind = 1;
mean = 0;
std = 0;
for i = 1 : length(samplePaths)
    spath = ['E:\newdata\',samplePaths(i).name];
    mpath = ['E:\newdata\',maskPaths(i).name];
    sampleFiles = dir(fullfile(spath, '*.dcm'));
    maskFiles = dir(fullfile(mpath, '*.jpg'));
    assert(length(sampleFiles) == length(maskFiles), 'sample and mask not equal');
    fprintf('dir is %s\n', spath);
    for j = 1 : length(sampleFiles)
        fprintf('Ind is %d\n', ind);
        [sfile, map] = dicomread([spath, '\', sampleFiles(j).name]);
        sfile = single(sfile);
        mfile = imread([mpath, '\', maskFiles(j).name]);
        mfile = imfill(mfile);
        mfile(mfile < 100) = 0;
        mfile(mfile > 100) = 1;
        mean = mean2(sfile);
        std = std2(sfile);
        sfile = sfile - mean;
        sfile = sfile / std;
        save(['E:\imgmats2\', num2str(ind, '%05d'), '.mat'], 'sfile');
        save(['E:\lblmats2\', num2str(ind, '%05d'), '.mat'], 'mfile');
        ind  = ind + 1;
        break;
    end
    break;
end

% path = 'E:\newdata\';
% 
% 
% 
% 
% clear;
% load('E:\newdata\newV1');
% resV = V1;
% [~, ~, num] = size(resV);
% path = 'CT070res\';
% if ~exist(path, 'dir')
%     mkdir(path);
% end
% figure(1);
% for i = num:-1:1
%     imwrite(resV(:, :, i), [path, num2str(num - i + 1, '%05d'), '.jpg']);
%     imshow(resV(:, :, i));
%     pause(0.01);
% end


% img = load('C:\Users\wwj\Desktop\pytorch-fcn\imgmats\00001.mat');
% img = img.sfile;
% subplot(1, 2, 1);
% imshow(img, []);
% subplot(1, 2, 2);
% imshow(downsample(downsample(img, 2)', 2)', []);
% pause(0.1);