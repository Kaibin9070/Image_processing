%% load original ultrasound image
clear
clc
% input  : png
% output : img
img4 = zeros(768, 1024, 3, 500, 'uint8'); 
img = zeros(768, 1024, 500);
image_dir = './output_png/';

for i = 1:500
    image_filename = fullfile(image_dir, sprintf('frame_%d.png', i));
    
    img4(:, :, :, i) = imread(image_filename);
    img(:, :, i) = img4(:, :, 1, i);
end

%% image registion and alignment
% input  : img
% output : regisImg
base = img(:,:,1); % frame_1 作為參考影像
fixed = base(131:400,381:600);
[optimizer, metric] = imregconfig('multimodal');
optimizer.InitialRadius = 0.009;
optimizer.Epsilon = 1.5e-4;
optimizer.GrowthFactor = 1.01;
optimizer.MaximumIterations = 300;

regisImg(:,:,1) = fixed;

for i = 2:size(img, 3)
    temp = img(:,:,i);
    moving = temp(131:400,381:600); % 150:330,380:600
    regisImg(:,:,i) = imregister(moving, fixed, 'translation', optimizer, metric);
    disp(['Iteration : ', num2str(i)]);
end

%% show and save registration result
reg_image_dir = './registered_image/';
for i = 1:size(regisImg, 3)
    imshow(regisImg(:,:,i), []); 
    title(['Aligned Image ', num2str(i)]); 
    drawnow; 
    outputFileName = fullfile(reg_image_dir, sprintf('regis_%d.png', i));
    imwrite(uint8(regisImg(:,:,i)), outputFileName); 
end
clc
%% denoise and image enhancement
% input  : registered image
% output : binarized image
% load registered image
reg_image_dir = './registered_image/';
bin_image_dir = './binarized_image/';
for i = 1:500
    reg_image_filename = fullfile(reg_image_dir, sprintf('regis_%d.png', i));
    
    regisImg(:,:,i) = imread(reg_image_filename);
end
% morphology component
line70 = strel('line',24,70);
line135 = strel('line',18,135);
circle3 = strel('disk', 3); 
[w,t,d] = size(regisImg);
enhanceImg = zeros(w, t, d);
for i = 1:d
    temp = mat2gray(regisImg(:,:,i));

    % denoise by median filter
    medtemp = imgaussfilt(temp, 2); % 1.5: denoise rate

    % enhancement
    gamma = 1.3;
    gammatemp = medtemp.^gamma;
    bw = imbinarize(gammatemp,0.25);
    bw1 = imfill(bw,'hole');
    bw2 = imclose(bw1,line70); % morphology
    bw3 = imclose(bw2,line135);

    enhanceImg(:,:,i) = imclose(bw3,circle3);

    % output
    outputFileName = fullfile(bin_image_dir, sprintf('bin_%d.png', i));
    imwrite(enhanceImg(:,:,i), outputFileName); 
end
clc
%% segmentation
% input  : binarized image
% output : roi image
%load binarized image
bin_image_dir = './binarized_image/';
roi_image_dir = './ROI/';
for i = 1:500
    bin_image_filename = fullfile(bin_image_dir, sprintf('bin_%d.png', i));
    enhanceImg(:,:,i) = imread(bin_image_filename);
end

% segmentation
imshow(enhanceImg(:,:,1));
%drawnow; 
disp('using click to label ROI (2 times)');
[xv,yv] = ginput(2);
m = zeros(size(enhanceImg,1), size(enhanceImg,2));
seg = zeros(size(enhanceImg,1), size(enhanceImg,2), size(enhanceImg,3));
m(yv(1):yv(2),xv(1):xv(2)) = 1;
itr = 240;
for i = 1:size(enhanceImg,3)
    seg(:,:,i) = region_seg(enhanceImg(:,:,i),m,itr);
    disp("The Region Based Active Contour Segmentation is " + int2str(i));
    disp("The Region Segment Interation is " + int2str(itr));

    % output
    outputFileName = fullfile(roi_image_dir, sprintf('roi_%d.png', i));
    imwrite(seg(:,:,i), outputFileName); 
end

%% calculate lumen diameter and  plot curve
% input  : roi image (seg)
% output : mathematic analysis
% load roi image
roi_image_dir = './ROI/';
for i = 1:500
    roi_image_filename = fullfile(roi_image_dir, sprintf('roi_%d.png', i));
    seg(:,:,i) = imread(roi_image_filename);
end

% calculate
length = size(seg,1)/3;
for i =1:size(seg,3)
    bound_seg = bwperim(seg(:,:,i)); 
    [row,col] = find(bound_seg>0);
    matrix = [row col]; 
    [eigenVector]=pca(matrix); %princomp
    transMatrix = eigenVector(:,end);
    Project_matrix = matrix * transMatrix;
    diameter(i) = ((max(Project_matrix)-min(Project_matrix))/length)*10;
end

curved = filloutliers(diameter,'linear');
plot(curved,'LineWidth',2);
title('Diameter');
ylabel('(mm)');

%% bonus: calculate ABI
[frame_peak,pk]=findvalley(curved,'q',21); 
[frame_valley,v] = findvalley(curved,'v',17);


avg_peak = mean(pk);
avg_valley = mean(v);
disp("avg_peak: "+ avg_peak );
disp("avg_valley: "+ avg_valley );
Ps = 101;
Pd = 77;
hd_index = log(Ps/Pd)/((avg_peak-avg_valley)/avg_valley);
disp("ABI: "+ hd_index);

% show
hold on;
findvalley(curved,'q',21); % find peak
findvalley(curved,'v',17); % find valley
title('Diameter');
ylabel('(mm)');
