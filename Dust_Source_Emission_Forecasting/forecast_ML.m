%% Step 1: Define ROI
clc, clear all, close all

Pixel_Resize=1         % 1 = 1km, 2 = km, ...
% Training_Rate = 0.01     % 0.4 = 40% for train and 60% for validation


% Read shapefile and field of year and month
roi = shaperead('polygon1kmfinal.shp');
merge_field = 'month'; % field of date in shapefile
all_field_vals = {roi(:).(merge_field)};
unique_field_vals = unique(all_field_vals);


c=1;
for k=1:numel(unique_field_vals)
   idx = find(ismember(all_field_vals, unique_field_vals{k}));
   x = roi(idx(1)).X;
   y = roi(idx(1)).Y;
   for i=2:numel(idx)
   [x,y] = polybool('union',x,y,roi(idx(i)).X,roi(idx(i)).Y);
   end
   s2(c).X = x;
   s2(c).Y = y;
   s2(c).Geometry = roi(idx(1)).Geometry;
   s2(c).(merge_field) = roi(idx(1)).(merge_field);
   c=c+1;
end

% Read Image for define the pixels located inside of ROI
 
[Image,aR] = geotiffread('aod2000-2021.tif');
NewPixSize = geotiffinfo('aod2000-2021.tif'); % Reread the resized image
[x,y] = pixcenters(NewPixSize); 
clear Image
% Convert x,y arrays to grid: 
[X,Y] = meshgrid(x,y);

% convret the date of data to be used for specify the ROI for each month
Real_date=(str2double(unique_field_vals))-100

% Stack the layer of monthly dust 
for i=1:numel(s2)
    j=Real_date(i)
 sizea=size(s2(i).X)
 h=((sizea(1, 2)+1)/6)
 for k=1:h
  maskday(:,:,k)= inpolygon(X,Y,(s2(i).X(((k-1)*6)+1:((k-1)*6)+4)),(s2(i).Y(((k-1)*6)+1:((k-1)*6)+4)));
    end
 mask(:,:,j)=sum(maskday, 3);
   clear maskday
end
mask(:,:,264)=0;
% define all of the pixels that record the dust for whole period
sum_mask=sum(mask, 3);
DustyPixlel=sum_mask;
% Reshape the ground truth data to a vector containing class labels
ReDustyPixlel=reshape(DustyPixlel, [], 1);
% Find the location indices of the ground truth vector that contain class labels. Discard labels with the value 0, as they are unlabeled and do not represent a class
gtLocs = find(ReDustyPixlel~=0);
gtLocs1 = ReDustyPixlel(gtLocs);

clear aR Resize_Image Real_date ry rx sum_mask NewPixSize ReDustyPixlel DustyPixlel y1 y2 y3 idx j h k s i X Y x y idx mask1 s2 sizeImage roi unique_field_vals all_field_vals merge_field

%% Step 2: Prepate input for classifiers
load("NDVI_2000_2021.mat","NDVI") 
% Resize driver data 

%NDVI
ReNDVI=m2_filter(NDVI, Pixel_Resize,  Pixel_Resize);

%Rainfall
load("Rainfall_2000_2021.mat","Rainfall") 
y1 = resample(double(Rainfall),1,Pixel_Resize);
y2=pagetranspose(y1);
y3 = resample(y2,1,Pixel_Resize);
ReRainfall=pagetranspose(y3);
clear y1 y2 y3 Rainfall
%SM
load("SM_2000_2021.mat","SM") 
y1 = resample(double(SM),1,Pixel_Resize);
y2=pagetranspose(y1);
y3 = resample(y2,1,Pixel_Resize);
ReSM=pagetranspose(y3);
clear y1 y2 y3 SM_2000_2021
%VS
load("ws_2000_2020.mat","VS") 
y1 = resample(double(VS),1,Pixel_Resize);
y2=pagetranspose(y1);
y3 = resample(y2,1,Pixel_Resize);
ReVS=pagetranspose(y3);
clear y1 y2 y3 VS

% clay
load("data2000-2021.mat","clay") 
y1 = resample(double(clay),1,Pixel_Resize);
y2=pagetranspose(y1);
y3 = resample(y2,1,Pixel_Resize);
ReClay=pagetranspose(y3);
clear y1 y2 y3 clay

% silt
load("data2000-2021.mat","silt") 
y1 = resample(double(silt),1,Pixel_Resize);
y2=pagetranspose(y1);
y3 = resample(y2,1,Pixel_Resize);
ReSilt=pagetranspose(y3);
clear y1 y2 y3 silt

% sand
load("data2000-2021.mat","sand") 
y1 = resample(double(sand),1,Pixel_Resize);
y2=pagetranspose(y1);
y3 = resample(y2,1,Pixel_Resize);
ReSand=pagetranspose(y3);
clear y1 y2 y3 sand

% soithickness
load("data2000-2021.mat","soithickness") 
y1 = resample(double(soithickness),1,Pixel_Resize);
y2=pagetranspose(y1);
y3 = resample(y2,1,Pixel_Resize);
ReSoithickness=pagetranspose(y3);
clear y1 y2 y3 soithickness

% DEM
load("data2000-2021.mat","DEM") 
y1 = resample(double(DEM),1,Pixel_Resize);
y2=pagetranspose(y1);
y3 = resample(y2,1,Pixel_Resize);
ReDEM=pagetranspose(y3);
clear y1 y2 y3 DEM


% Slope
load("data2000-2021.mat","Slope") 
y1 = resample(double(Slope),1,Pixel_Resize);
y2=pagetranspose(y1);
y3 = resample(y2,1,Pixel_Resize);
ReSlope=pagetranspose(y3);
clear y1 y2 y3 Slope

% LST
load("LST_2000_2021.mat","LST")
y1 = resample(double(LST),1,Pixel_Resize);
y2=pagetranspose(y1);
y3 = resample(y2,1,Pixel_Resize);
ReLST=pagetranspose(y3);
clear y1 y2 y3 LST

bandImageFiltered_Clay = imgaussfilt(ReClay,2);
bandImageGray_Clay = mat2gray(bandImageFiltered_Clay);
bandImageGray_Clay = mat2gray(bandImageFiltered_Clay);
hsDataFiltered_Clay = uint8(bandImageGray_Clay*255);

bandImageFiltered_Silt = imgaussfilt(ReSilt,2);
bandImageGray_Silt = mat2gray(bandImageFiltered_Silt);
bandImageGray_Silt = mat2gray(bandImageFiltered_Silt);
hsDataFiltered_Silt = uint8(bandImageGray_Silt*255);

bandImageFiltered_Sand = imgaussfilt(ReSand,2);
bandImageGray_Sand = mat2gray(bandImageFiltered_Sand);
bandImageGray_Sand = mat2gray(bandImageFiltered_Sand);
hsDataFiltered_Sand = uint8(bandImageGray_Sand*255);

bandImageFiltered_Soithickness = imgaussfilt(ReSoithickness,2);
bandImageGray_Soithickness = mat2gray(bandImageFiltered_Soithickness);
bandImageGray_Soithickness = mat2gray(bandImageFiltered_Soithickness);
hsDataFiltered_Soithickness = uint8(bandImageGray_Soithickness*255);

bandImageFiltered_Slope = imgaussfilt(ReSlope,2);
bandImageGray_Slope = mat2gray(bandImageFiltered_Slope);
bandImageGray_Slope = mat2gray(bandImageFiltered_Slope);
hsDataFiltered_Slope = uint8(bandImageGray_Slope*255);

bandImageFiltered_DEM = imgaussfilt(ReDEM,2);
bandImageGray_DEM = mat2gray(bandImageFiltered_DEM);
bandImageGray_DEM = mat2gray(bandImageFiltered_DEM);
hsDataFiltered_DEM = uint8(bandImageGray_DEM*255);

% Apply a Gaussian filter (σ=2) to each month data using the imgaussfilt function, and then convert them to grayscale images.
[M,N,C] = size(ReLST);
hsDataFiltered_LST = zeros(size(ReLST));
hsDataFiltered_NDVI = zeros(size(ReNDVI));
hsDataFiltered_Rainfall = zeros(size(ReRainfall));
hsDataFiltered_SM = zeros(size(ReSM));
hsDataFiltered_VS = zeros(size(ReVS));

for band = 1:C
    LST_1 = ReLST(:,:,band); 
    NDVI_1 = ReNDVI(:,:,band); 
    Rainfall_1 = ReRainfall(:,:,band); 
    SM_1 = ReSM(:,:,band); 
    VS_1 = ReVS(:,:,band);
    
    bandImageFiltered_LST = imgaussfilt(LST_1,2);
    bandImageFiltered_NDVI = imgaussfilt(NDVI_1,2); 
    bandImageFiltered_Rainfall = imgaussfilt(Rainfall_1,2); 
    bandImageFiltered_SM = imgaussfilt(SM_1,2);
    bandImageFiltered_VS = imgaussfilt(VS_1,2);

    bandImageGray_LST = mat2gray(bandImageFiltered_LST);
    bandImageGray_NDVI = mat2gray(bandImageFiltered_NDVI);
    bandImageGray_Rainfall = mat2gray(bandImageFiltered_Rainfall);
    bandImageGray_SM = mat2gray(bandImageFiltered_SM);
    bandImageGray_VS = mat2gray(bandImageFiltered_VS);
    
    hsDataFiltered_LST(:,:,band) = uint8(bandImageGray_LST*255);
    hsDataFiltered_NDVI(:,:,band) = uint8(bandImageGray_NDVI*255);
    hsDataFiltered_Rainfall(:,:,band) = uint8(bandImageGray_Rainfall*255);
    hsDataFiltered_SM(:,:,band) = uint8(bandImageGray_SM*255);
    hsDataFiltered_VS(:,:,band) = uint8(bandImageGray_VS*255);
    
end

% Reshape the filtered data to a set of feature vectors
DataVector_LST = reshape(hsDataFiltered_LST,[M*N C]);
DataVector_NDVI = reshape(hsDataFiltered_NDVI,[M*N C]);
DataVector_Rainfall = reshape(hsDataFiltered_Rainfall,[M*N C]);
DataVector_SM = reshape(hsDataFiltered_SM,[M*N C]);
DataVector_VS = reshape(hsDataFiltered_VS,[M*N C]);

DataVector_Clay=reshape(hsDataFiltered_Clay, [], 1); % Single band reshape
DataVector_Silt=reshape(hsDataFiltered_Silt, [], 1);
DataVector_Sand=reshape(hsDataFiltered_Sand, [], 1);
DataVector_Soithickness=reshape(hsDataFiltered_Soithickness, [], 1);
DataVector_DEM=reshape(hsDataFiltered_DEM, [], 1);
DataVector_Slope=reshape(hsDataFiltered_Slope, [], 1);



DataVector_DEM1=reshape(ReDEM, [], 1);




DataVector_mask = reshape(mask,[M*N C]); % ROI reshape


% Sepreat dust driver data specefic monnth of year
LST1=(DataVector_LST);
NDVI1=(DataVector_NDVI);
Rainfall1=(DataVector_Rainfall);
SM1=(DataVector_SM);
VS1=(DataVector_VS);

mask1=(DataVector_mask);

for i=1:264

% Choose the data just for afor-recorded as dust center pixel
LST2(:, i)=LST1(gtLocs, i);
NDVI2(:, i)=NDVI1(gtLocs, i);
Rainfall2(:, i)=Rainfall1(gtLocs, i);
SM2(:, i)=SM1(gtLocs, i);
VS2(:, i)=VS1(gtLocs, i);

Clay2(:, i)=DataVector_Clay(gtLocs, 1);
Silt2(:, i)=DataVector_Silt(gtLocs, 1);
Sand2(:, i)=DataVector_Sand(gtLocs, 1);
Soithickness2(:, i)=DataVector_Soithickness(gtLocs, 1);
DEM2(:, i)=DataVector_DEM(gtLocs, 1);
Slope2(:, i)=DataVector_Slope(gtLocs, 1);


end

for i=1:264
DEM3(:, i)=DataVector_DEM1(gtLocs, 1);

mask2(:, i)=mask1(gtLocs, i);

end
NDVI_Mean=nanmean(NDVI2, 1);
LST_Mean=nanmean(LST2, 1);
SM_Mean=nanmean(SM2, 1);
VS_Mean=nanmean(VS2, 1);
Rainfall_Mean=nanmean(Rainfall2, 1);

% put all year driver data togather
DataVector (1, :)=reshape(LST2(:, :), [], 1);
DataVector (2, :)=reshape(NDVI2(:, :), [], 1);
DataVector (3, :)=reshape(Rainfall2(:, :), [], 1);
DataVector (4, :)=reshape(SM2(:, :), [], 1);
DataVector (5, :)=reshape(VS2(:, :), [], 1);
DataVector (6, :)=reshape(Clay2(:, :), [], 1);
DataVector (7, :)=reshape(Silt2(:, :), [], 1);
DataVector (8, :)=reshape(Sand2(:, :), [], 1);
DataVector (9, :)=reshape(Soithickness2(:, :), [], 1);
DataVector (10, :)=reshape(DEM2(:, :), [], 1);
DataVector (11, :)=reshape(Slope2(:, :), [], 1);

DataVector3 (1, :)=reshape(DEM3(:, :), [], 1);

DataVector2000_2020 (1, :)=reshape(LST2(:, 1:252), [], 1);
DataVector2000_2020 (2, :)=reshape(NDVI2(:, 1:252), [], 1);
DataVector2000_2020 (3, :)=reshape(Rainfall2(:, 1:252), [], 1);
DataVector2000_2020 (4, :)=reshape(SM2(:, 1:252), [], 1);
DataVector2000_2020 (5, :)=reshape(VS2(:, 1:252), [], 1);
DataVector2000_2020 (6, :)=reshape(Clay2(:, 1:252), [], 1);
DataVector2000_2020 (7, :)=reshape(Silt2(:, 1:252), [], 1);
DataVector2000_2020 (8, :)=reshape(Sand2(:, 1:252), [], 1);
DataVector2000_2020 (9, :)=reshape(Soithickness2(:, 1:252), [], 1);
DataVector2000_2020 (10, :)=reshape(DEM2(:, 1:252), [], 1);
DataVector2000_2020 (11, :)=reshape(Slope2(:, 1:252), [], 1);

DataVector2021 (1, :)=reshape(LST2(:, 253:264), [], 1);
DataVector2021 (2, :)=reshape(NDVI2(:, 253:264), [], 1);
DataVector2021 (3, :)=reshape(Rainfall2(:, 253:264), [], 1);
DataVector2021 (4, :)=reshape(SM2(:, 253:264), [], 1);
DataVector2021 (5, :)=reshape(VS2(:, 253:264), [], 1);
DataVector2021 (6, :)=reshape(Clay2(:, 253:264), [], 1);
DataVector2021 (7, :)=reshape(Silt2(:, 253:264), [], 1);
DataVector2021 (8, :)=reshape(Sand2(:, 253:264), [], 1);
DataVector2021 (9, :)=reshape(Soithickness2(:, 253:264), [], 1);
DataVector2021 (10, :)=reshape(DEM2(:, 253:264), [], 1);
DataVector2021 (11, :)=reshape(Slope2(:, 253:264), [], 1);


DataVectororiginal=DataVector;
gtVector=reshape(mask2(:, :), [], 1);
DataVector2000_2020=DataVector2000_2020';
DataVector2021=DataVector2021';
gtVector2000_2020 =reshape(mask2(:, 1:252), [], 1);
gtVector2000_2020=gtVector2000_2020';
gtVector2021 =reshape(mask2(:, 253:264), [], 1);

gtVector =reshape(mask2(:, :), [], 1);


size_DataVector2000_2020=size(DataVector2000_2020)
size_DataVector=size(DataVector)
size_DataVector2021=size(DataVector2021)

gt11 = find(gtVector~=0);

sizegtlocs=size(gtLocs)

% LULC
[Image,aR] = geotiffread('LULC_TEB.tif');
LULC(:,:,2:22)=Image(:,1:end-1,:);
LULC(:,:,1)=Image(:,1:end-1,1);
DataVector_LULC = reshape(LULC,[M*N 22]);
LULC2=reshape(LULC(:,:,22), [], 1);
for i=1:12
% Choose the data just for afor-recorded as dust center pixel

LULC3(:, i)=LULC2(gtLocs, 1);

end
LULC4=reshape(LULC3, [], 1);

DataVector=DataVector';
% balance sample by SMOTE
Ratio2000_2020=floor(numel(find(gtVector2000_2020==0))/numel(find(gtVector2000_2020==1)));
DataVector2000_2020_1=DataVector2000_2020(find(gtVector2000_2020==1), :);
gtVector2000_2020_1=gtVector2000_2020(find(gtVector2000_2020==1));
size_gtVector2000_2020_1=size(gtVector2000_2020_1)

Ratio=floor(numel(find(gtVector==0))/numel(find(gtVector==1)));
DataVector1=DataVector(find(gtVector==1), :);
gtVector1=gtVector(find(gtVector==1));
size_gtVector1=size(gtVector1)

% if Ratio<size_gtVector1(1, 1)
K2000_2020=floor(Ratio2000_2020/2) % (Ratio/1) = equal 0 and 1,(Ratio/3) = 67.7% 0 and 33.3% 1, ... 
[X2000_2020,C2000_2020,Xn2000_2020,Cn2000_2020]=smote(DataVector2000_2020_1,K2000_2020, K2000_2020+1,'Class', gtVector2000_2020_1);

K=floor(Ratio/2) % (Ratio/1) = equal 0 and 1,(Ratio/3) = 67.7% 0 and 33.3% 1, ... 
[X,C,Xn,Cn]=smote(DataVector1,K, K+1,'Class', gtVector1);


gtVector2000_2020=gtVector2000_2020';
size_newSampleY2000_2020=size(Xn2000_2020)
DataVector2000_2020(size_DataVector2000_2020(1, 1)+1:size_DataVector2000_2020(1, 1)+size_newSampleY2000_2020(1, 1), :)=Xn2000_2020;
gtVector2000_2020(size_DataVector2000_2020(1, 1)+1:size_DataVector2000_2020(1, 1)+size_newSampleY2000_2020(1, 1), 1)=Cn2000_2020;
gtTotal2000_2020 = find(gtVector2000_2020~=-1);
gt2000_2020_1 = find(gtVector2000_2020~=0);

size_newSampleY=size(Xn)
DataVector(size_DataVector(1, 1)+1:size_DataVector(1, 1)+size_newSampleY(1, 1), :)=Xn;
gtVector(size_DataVector(1, 1)+1:size_DataVector(1, 1)+size_newSampleY(1, 1), 1)=Cn;
gtTotal = find(gtVector~=-1);
gt1 = find(gtVector~=0);
gtVector2021_1 = find(gtVector2021~=0);
gtLULC_1=LULC4(gtVector2021_1 );

clear K2000_2020 size_newSampleY2000_2020 Rainfall LST silt DEM Slope soithickness clay sand SM NDVI VS Rainfall1 LST1 clay1 Soithickness1 silt1 NDVI1 SM1VS1 Rainfall2 AOD2 Sand2 Silt2 Clay2 Silt2 SM2 Soithickness2 VS2 NDVI2 ...
    bandImageGray_LST bandImageGray_Clay bandImageGray_Silt bandImageGray_Sand bandImageGray_Soithickness bandImageGray_NDVI bandImageGray_VS bandImageGray_SM bandImageGray_Rainfall...
    hsDataFiltered_Clay hsDataFiltered_Silt hsDataFiltered_Sand hsDataFiltered_Soithickness hsDataFiltered_LST hsDataFiltered_NDVI hsDataFiltered_Rainfall hsDataFiltered_SM hsDataFiltered_VS...
    DataVector_Clay DataVector_Silt DataVector_Sand DataVector_Soithickness DataVector_LST DataVector_NDVI DataVector_Rainfall DataVector_SM DataVector_VS...
    ReRainfall ReLST ReSilt ReSand ReSoithickness ReClay ReSM ReNDVI ReVS Rainfall1 LST1  NDVI_1 SM_1 VS_1 LST_1 Rainfall_1 j i c M N C ...
    bandImageFiltered_LST bandImageFiltered_Clay bandImageFiltered_Silt bandImageFiltered_NDVI bandImageFiltered_Sand bandImageFiltered_Rainfall bandImageFiltered_VS bandImageFiltered_SM bandImageFiltered_Soithickness...
    SizeNew VS1 sizedata band DataVector_mask mask1 mask2 Image gtLocs DataVector1 Ratio gtVector1 size_newSampleY X C Xn Cn n 


%% Step 3: Forecasting and Validation

%%%%%%%%%%%%%%%%%%%%%%%%% Summary Statistics %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define the function for summary statistics
summary_stats = @(vec) [mean(vec), median(vec), std(vec), min(vec), max(vec)];

% We will obtain the number of variables from the data matrix directly
num_variables = size(DataVector, 2);

% Initialize an empty matrix for the summary statistics
statistics = zeros(num_variables, 5); % A row for each variable and a column for each statistic

% Generate variable names based on the number of variables
variables = arrayfun(@(i) ['Variable' num2str(i)], 1:num_variables, 'UniformOutput', false);

% Calculate summary statistics for each variable
for i = 1:num_variables
    vec = DataVector(:,i);  % Extract the i-th variable data as a vector
    statistics(i, :) = summary_stats(vec);  % Apply the function and store the results
end

% Convert the statistics matrix to a table
stat_names = {'Mean', 'Median', 'StandardDeviation', 'Min', 'Max'};
stats_table = array2table(statistics, 'VariableNames', stat_names, 'RowNames', variables);
disp(stats_table);
%%%%%%%%%%%%%%%%%%%%%%%%% Data Distribution %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
for i = 1:size(DataVector, 2)
    subplot(3, 4, i); % Adjust the grid size according to the number of variables
    histogram(DataVector(:,i));
    title(variables{i});
end
sgtitle('Distributions of Environmental Variables');



%%%%%%%%%%%%%%%%%%%%%%%%%  Correlations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
corrMatrix = corr(DataVector);

% Plot the correlations using a heatmap
figure;
imagesc(corrMatrix);
colorbar;
title('Correlation Heatmap');
set(gca, 'XTick', 1:numel(variables), 'XTickLabel', variables, 'YTick', 1:numel(variables), 'YTickLabel', variables);
xtickangle(45); 
axis square;


%%%%%%%%%%%%%%%%%%%%%%%%% Pairwise Relationships%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For large datasets, it might be computationally intensive to plot scatter plot for all pairs
% Consider using a subset or plotting only a few representative pairs

% Sample size
numSamples = 10000; % Adjust the number based on your dataset size
sampleIdx = randsample(size(DataVector, 1), numSamples);

% Pairwise scatter plots for a sample of the data
figure;
for i = 1:length(variables)
    for j = i+1:length(variables)
        subplot(length(variables)-1, length(variables)-1, (i-1)*(length(variables)-1)+j-1);
        scatter(DataVector(sampleIdx, i), DataVector(sampleIdx, j), '.');
        xlabel(variables{i});
        ylabel(variables{j});
    end
end

%%%%%%%%%%%%%%%%%%%%%%% Visualizing time-series data %%%%%%%%%%%%%%%%%%%%%%%%%
% Load 'VS' variable from the .mat file
load('data2000-2021.mat', 'VS');

% Check if 'VS' variable is loaded into your workspace.
if exist('VS', 'var')
    % Assuming your resizing function or script is correctly converting 'VS' to 'VS'
    ReVS; % replace 'resize_VS_function' with your actual function
else
    error('VS variable is not loaded properly.');
end


% divide dataset to training and validation sets
cv = cvpartition(gtVector,"HoldOut", 1-Training_Rate);
locTrain = gtTotal(cv.training);+
locValidation = gtTotal(~cv.training);

cv2000_2020 = cvpartition(gtVector2000_2020,"HoldOut", 1-Training_Rate);
locTrain2000_2020 = gtTotal2000_2020(cv2000_2020.training);
locValidation2000_2020 = gtTotal2000_2020(~cv2000_2020.training);

% RF random divide all year
RFMdl = TreeBagger(100,DataVector(locTrain,:),gtVector(locTrain,:),Weights=sampleWeights);
[~,RF_ROC_Smote] = predict(RFMdl,(DataVector(locValidation,:)));
% RF_ROC_SmoteRandon = cellfun(@str2num,RF_ROC_Smote);
[RF_OA_Smote, ~] = predict(RFMdl,(DataVector(locValidation,:)));
RF_OA_SmoteRand = cellfun(@str2num,RF_OA_Smote);
RF_OA_SmoteRandom=sum(RF_OA_SmoteRand==gtVector(locValidation))/size(gtVector(locValidation),1);
disp(["RF Real_Smote_01 OA= ",num2str(RF_OA_SmoteRandom)])

RFMdl = TreeBagger(100,DataVector(locTrain,:),gtVector(locTrain,:),Method="regression",Surrogate="on",OOBPredictorImportance="on");
importance = RFMdl.OOBPermutedPredictorDeltaError;
[~,idx] = sort(importance,'descend');
numFeatures = 2;
selectedIdx = idx(1:numFeatures);

RFMdl = fitrensemble(DataVector(locTrain,:),gtVector(locTrain,:));
[~,RF_ROC_Smote] = predict(RFMdl,(DataVector(locValidation,:)));
% RF_ROC_SmoteRandon = cellfun(@str2num,RF_ROC_Smote);
[RF_OA_Smote, ~] = predict(RFMdl,(DataVector(locValidation,:)));
RF_OA_SmoteRand = cellfun(@str2num,RF_OA_Smote);
RF_OA_SmoteRandom=sum(RF_OA_SmoteRand==gtVector(locValidation))/size(gtVector(locValidation),1);
disp(["RF Real_Smote_01 OA= ",num2str(RF_OA_SmoteRandom)])

impOOB = oobPermutedPredictorImportance(RFMdl);
oobErrorBaggedEnsemble = oobError(RFMdl);
plot(oobErrorBaggedEnsemble)
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';

rocRF_01_Smote = rocmetrics(gtVector(locValidation,:),RF_ROC_Smote,RFMdl.ClassNames);
rocRF_01_Smote.AUC
figure %Plot ROC
tiledlayout(1,1,TileSpacing="loose")
nexttile
plot(rocRF_01_Smote)
title("ROC RF For Smote samples")

%RF Trained by 2000-2020 test on 2021 for 0 and 1 classes
RFMdl2000_2020 = TreeBagger(100,DataVector2000_2020,gtVector2000_2020,'OOBPred','On');
[RF_OA_Real2021_01, ~] = predict(RFMdl2000_2020,DataVector2021);
RF_OA_Real2021_01Rand = cellfun(@str2num,RF_OA_Real2021_01);
[~, RF_ROC_Real2021_01] = predict(RFMdl2000_2020,DataVector2021);
RF_Real2021_OA=sum(RF_OA_Real2021_01Rand==gtVector2021)/size(gtVector2021,1);
disp(["RF Real 2021_01 OA= ",num2str(RF_Real2021_OA)])
rocRF_Real2021_01 = rocmetrics(gtVector2021,RF_ROC_Real2021_01,RFMdl2000_2020.ClassNames);
rocRF_Real2021_01.AUC
figure %Plot ROC
tiledlayout(1,1,TileSpacing="loose")
nexttile
plot(rocRF_Real2021_01)
title("ROC RF For Real2021 01 samples")
%%%%%%%%%%%

clasLULC=find(LULC4(RF_OA_Real2021_01Rand~=gtVector2021));
typeLULC=LULC4(clasLULC);



% Train the Random Forest with OOB Predictor Importance computation
RFMdl2000_2020 = TreeBagger(100, DataVector2000_2020, gtVector2000_2020, 'OOBPredictorImportance', 'on');

% View the importance of features
featureImportance = RFMdl2000_2020.OOBPermutedVarDeltaError;
bar(featureImportance);
xlabel('Feature Number');
ylabel('Out-Of-Bag Feature Importance');

% Plot ROC Curve
figure;
plot(X, Y);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve for RF Classifier on 2021 Data');
text(0.5,0.5,strcat('AUC = ', num2str(AUC)), 'FontSize', 14);
%

% SVM random divide all year
svmMdl = fitcsvm(DataVector(locTrain,:),gtVector(locTrain,:));
[~,SVM_ROC_Smote] = predict(svmMdl,(DataVector(locValidation,:)));
[SVM_OA_Smote, ~] = predict(svmMdl,(DataVector(locValidation,:)));
SVM_OA_Smote=sum(SVM_OA_Smote==gtVector(locValidation))/size(gtVector(locValidation),1);
disp(["SVM Real_Smote_01 OA= ",num2str(SVM_OA_Smote)])
rocSVM_01_Smote = rocmetrics(gtVector(locValidation,:),SVM_ROC_Smote,svmMdl.ClassNames);
rocSVM_01_Smote.AUC
figure %Plot ROC
tiledlayout(1,1,TileSpacing="loose")
nexttile
plot(rocSVM_01_Smote)
title("ROC SVM For Smote samples")

%SVM Trained by 2000-2020 test on 2021 for 0 and 1 classes
svmMdl2000_2020 = fitcsvm(DataVector2000_2020,gtVector2000_2020);
[SVM_OA_Real2021_01, ~] = predict(svmMdl2000_2020,DataVector2021);
[~, SVM_ROC_Real2021_01] = predict(svmMdl2000_2020,DataVector2021);
SVM_Real2021_OA=sum(SVM_OA_Real2021_01==gtVector2021)/size(gtVector2021,1);
disp(["SVM Real 2021_01 OA= ",num2str(SVM_Real2021_OA)])
rocSVM_Real2021_01 = rocmetrics(gtVector2021,SVM_ROC_Real2021_01,svmMdl2000_2020.ClassNames);
rocSVM_Real2021_01.AUC
figure %Plot ROC
tiledlayout(1,1,TileSpacing="loose")
nexttile
plot(rocSVM_Real2021_01)
title("ROC SVM For Real2021 01 samples")

%MNB Trained by 2000-2020test on 2021 for 0 and 1 classes
MNBMdl2000_2020 = fitcnb(DataVector2000_2020,gtVector2000_2020);
[MNB_OA_Real2021_01, ~] = predict(MNBMdl2000_2020,DataVector2021);
[~, MNB_ROC_Real2021_01] = predict(MNBMdl2000_2020,DataVector2021);
MNB_Real2021_OA=sum(MNB_OA_Real2021_01==gtVector2021)/size(gtVector2021,1);
disp(["MNB Real 2021_01 OA= ",num2str(MNB_Real2021_OA)])
rocMNB_Real2021_01 = rocmetrics(gtVector2021,MNB_ROC_Real2021_01,MNBMdl2000_2020.ClassNames);
rocMNB_Real2021_01.AUC
figure %Plot ROC
tiledlayout(1,1,TileSpacing="loose")
nexttile
plot(rocMNB_Real2021_01)
title("ROC MNB For Real2021 01 samples")

% KNN random divide all year
KNNMdl = fitcknn(DataVector(locTrain,:),gtVector(locTrain,:));
[~,KNN_ROC_Smote] = predict(KNNMdl,(DataVector(locValidation,:)));
[KNN_OA_Smote, ~] = predict(KNNMdl,(DataVector(locValidation,:)));
KNN_OA_Smote=sum(KNN_OA_Smote==gtVector(locValidation))/size(gtVector(locValidation),1);
disp(["KNN Real_Smote_01 OA= ",num2str(KNN_OA_Smote)])
rocKNN_01_Smote = rocmetrics(gtVector(locValidation,:),KNN_ROC_Smote,KNNMdl.ClassNames);
rocKNN_01_Smote.AUC
figure %Plot ROC
tiledlayout(1,1,TileSpacing="loose")
nexttile
plot(rocKNN_01_Smote)
title("ROC KNN For Smote samples")

%KNN Trained by 2000-2020 test on 2021 for 0 and 1 classes
KNNMdl2000_2020 = fitcknn(DataVector2000_2020,gtVector2000_2020);
[KNN_OA_Real2021_01, ~] = predict(KNNMdl2000_2020,DataVector2021);
[~, KNN_ROC_Real2021_01] = predict(KNNMdl2000_2020,DataVector2021);
KNN_Real2021_OA=sum(KNN_OA_Real2021_01==gtVector2021)/size(gtVector2021,1);
disp(["KNN Real 2021_01 OA= ",num2str(KNN_Real2021_OA)])
rocKNN_Real2021_01 = rocmetrics(gtVector2021,KNN_ROC_Real2021_01,KNNMdl2000_202.ClassNames);
rocKNN_Real2021_01.AUC
figure %Plot ROC
tiledlayout(1,1,TileSpacing="loose")
nexttile
plot(rocKNN_Real2021_01)
title("ROC KNN For Real2021 01 samples")



clear KNN_ROC_Smote KNN_OA_Smote KNN_ROC_Real2021_01 KNN_Real2021_OA KNN_OA_Real1 KNN_ROC_Real2021_1 KNN_Real_OA_1numFeatures classes XTrain XValidation filterSize numFilters MiniBatchSize options YPred MNBLabelOut1 MNBLabelOut KNNLabelOut1 KNNLabelOut1...
    outputs_RF RFLabelOut1 svmLabelOut svmLabelOut1 locTrain locValidation 


% Assuming your full dataset is in DataVector2021 and trainedClassifier exists.

% Set up perturbation parameters
gtCategorical = categorical(gtVector2021);
RFMdl2000_2020 = TreeBagger(100,DataVector2021,gtCategorical,'OOBPred','On');
numFeatures = size(DataVector2021, 2);
numPerturbations = 100;  % Number of perturbations per feature
perturbationRange = linspace(-0.2, 0.2, numPerturbations);  % e.g., +/- 10% for each feature

% Preallocate array to store sensitivity information
sensitivityAnalysis = zeros(numFeatures, numPerturbations);

for i = 1:numFeatures  % Iterate over all features
    for j = 1:numPerturbations  % Iterate over perturbations for each feature
        % Create a perturbed copy of your test data
        perturbedData = DataVector2021;
        
        % Apply perturbation to the i-th feature
        perturbedData(:, i) = perturbedData(:, i) + perturbationRange(j) * perturbedData(:, i);
        
        % Make predictions with perturbed data
        perturbedPredictions = predict(RFMdl2000_2020, perturbedData);
        
        % Convert cell arrays to numeric arrays
        perturbedPredictionsNumeric = str2double(perturbedPredictions);
        gtCategoricalNumeric = double(gtCategorical);
        
        % In this case, measure sensitivity as the change in predictions
        sensitivityAnalysis(i, j) = mean(abs(perturbedPredictionsNumeric - gtCategoricalNumeric));
    end
end
% Now you can visualize the sensitivity for each feature
for i = 1:numFeatures
    figure;
    plot(perturbationRange, sensitivityAnalysis(i, :), '-o');
    xlabel('Perturbation Range');
    ylabel('Average Change in Prediction');
    title(['Sensitivity Analysis for Feature ', num2str(i)]);
    grid on;
end


% Assuming your full dataset is in DataVector2021 and trainedClassifier is a trained Random Forest model.

% Set up perturbation parameters
numFeatures = size(DataVector2021, 2);
numPerturbations = 200;  % Number of perturbations per feature
perturbationRange = linspace(0, 1, numPerturbations);  % e.g., +/- 10% for each feature

% Preallocate array to store sensitivity information
sensitivityAnalysis = zeros(numFeatures, numPerturbations);

for i = 1:numFeatures  % Iterate over all features
    for j = 1:numPerturbations  % Iterate over perturbations for each feature
        % Create a perturbed copy of your test data
        perturbedData = DataVector2021;
        
        % Apply perturbation to the i-th feature
        perturbedData(:, i) = perturbedData(:, i) + perturbationRange(j) * perturbedData(:, i);
        
        % Make predictions with perturbed data using Random Forest model
        [labels, scores] = predict(trainedClassifier, perturbedData); % 'labels' are categorical
        perturbedPredictions = str2double(labels);  % Convert categorical labels to numeric if necessary
        
        % In this case, measure sensitivity as the change in predictions
        sensitivityAnalysis(i, j) = mean(abs(perturbedPredictions - YTest));
    end
end

% Now you can visualize the sensitivity for each feature
for i = 1:numFeatures
    figure;
    plot(perturbationRange, sensitivityAnalysis(i, :), '-o');
    xlabel('Perturbation Range');
    ylabel('Average Change in Prediction');
    title(['Sensitivity Analysis for Feature ', num2str(i)]);
    grid on;
end

% Assuming you have your data stored in DataVector2000_2020 and
% gtVector2000_2020

% Convert gtVector2000_2020 from cell array of strings to categorical array
gtCategorical = categorical(gtVector2000_2020);

% Create a Random Forest Classifier model
model = TreeBagger(100, DataVector2000_2020, gtCategorical);

% Perform permutation importance
numFeatures = size(DataVector2000_2020, 2);
numIterations = 10;
importances = zeros(numFeatures, 1);

for i = 1:numIterations
    permutedData = DataVector2000_2020;
    for j = 1:numFeatures
        % Permute the j-th feature
        permutedData(:, j) = permutedData(randperm(size(DataVector2000_2020, 1)), j);
    end
    
    % Predict using permuted data
    predictions = predict(model, permutedData);
    
    % Calculate accuracy with permuted data
    accuracy = sum(predictions == gtCategorical) / numel(gtCategorical);
    
    % Calculate feature importance
    importances = importances + (1 - accuracy);
end

% Average the importances over iterations
importances = importances / numIterations;

% Sort the feature importances in descending order
[sortedImportances, sortedIndices] = sort(importances, 'descend');

% Print the feature importances
for i = 1:numFeatures
    fprintf('Feature %d: Importance = %.4f\n', sortedIndices(i), sortedImportances(i));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% RANDOM FOREST OPTIMAZATION  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define additional hyperparameter optimization variables
maxNumSplits = optimizableVariable('MaxNumSplits', [10, 100], 'Type', 'integer');
minLeafSize = optimizableVariable('MinLeafSize', [1, 60], 'Type', 'integer');
maxFeatures = optimizableVariable('MaxFeatures', [1, size(DataVector2000_2020, 2)], 'Type', 'integer');

% Optimize the hyperparameters using Bayesian optimization
fun = @(x) myObjectiveFunction(x, DataVector2000_2020, gtVector2000_2020);                           
results = bayesopt(fun, [numTrees, maxNumSplits, minLeafSize, maxFeatures], ...
    'Verbose', 1, 'AcquisitionFunctionName', 'expected-improvement-plus');

% Get the optimal hyperparameters
opt = results.XAtMinObjective;

% Create a template for tree learners specifying the MinLeafSize and NumVariablesToSample
% Ensure that NumVariablesToSample does not exceed the number of features you're using
N = 10;
maxVars = min(N, opt.MaxFeatures);
t = templateTree('MinLeafSize', opt.MinLeafSize, 'NumVariablesToSample', maxVars);

% Train the initial Random Forest model with the optimal hyperparameters to estimate predictor importance
initialRFModel = fitrensemble(DataVector2000_2020, gtVector2000_2020, ...
    'Method', 'Bag', ...
    'NumLearningCycles', opt.numTrees, ...
    'Learners', t);  % Use the template here

% Estimate the predictor importance
imp = predictorImportance(initialRFModel);

% Assuming 'imp' holds the importance scores from predictorImportance
[sortedImp, sortedIdx] = sort(imp, 'descend');
% Select the top N features based on importance
N = 10; % Set N to the number of features you wish to keep
topFeaturesIdx = sortedIdx(1:N);
topFeaturesData = DataVector2000_2020(:, topFeaturesIdx);

% Now retrain the Random Forest model using only the top N features
finalRFModel = fitrensemble(topFeaturesData, gtVector2000_2020, ...
    'Method', 'Bag', ...
    'NumLearningCycles', opt.numTrees, ...
    'Learners', t);  % Use the template here

DataVector2021_TopFeatures = DataVector2021(:, topFeaturesIdx);

% Predict the responses for 2021 data using the selected top features
CE_OA_Real2021_01 = predict(finalRFModel, DataVector2021_TopFeatures);
[~, CE_ROC_Real2021_01] = predict(finalRFModel, DataVector2021_TopFeatures);
CE_Real2021_OA = sum(CE_OA_Real2021_01 == gtVector2021) / size(gtVector2021, 1);
disp(["CE Real 2021_01 OA= ", num2str(CE_Real2021_OA)])
rocCE_Real2021_01 = rocmetrics(gtVector2021, CE_ROC_Real2021_01, finalRFModel.ClassNames);
rocCE_Real2021_01.AUC
figure % Plot ROC
tiledlayout(1,1,TileSpacing="loose")
nexttile
plot(rocCE_Real2021_01)
title("ROC CE For Real2021 01 samples")



% Incorrect usage if 'model' is not defined
predictedResponses = predict(model, DataVector2021_TopFeatures);

% Correct usage assuming 'finalRFModel' is your trained model
predictedResponses = predict(finalRFModel, DataVector2021_TopFeatures);



%%%%%%%%%%%%%%%%%%%%%%%%
% Define the hyperparameter optimization variables
numTrees = optimizableVariable('numTrees',[10,200],'Type','integer');

% Define the objective function
fun = @(x) myObjectiveFunction(x,DataVector2000_2020, gtVector2000_2020);

% Optimize the hyperparameters using Bayesian optimization
results = bayesopt(fun, numTrees, 'Verbose', 1, 'AcquisitionFunctionName', 'expected-improvement-plus');

% Train a random forest model using the optimal hyperparameters
opt = results.XAtMinObjective;
RFMddl2000_2020 = fitrensemble(DataVector2000_2020, gtVector2000_2020, ...
    'Method', 'Bag', 'NumLearningCycles', opt.numTrees);


% Train a classification ensemble using AdaBoost and decision stumps
CEMdl2000_2020 = fitcensemble(DataVector2000_2020, gtVector2000_2020, ...
    'Method', 'AdaBoostM1', 'Learners', 'tree', 'NumLearningCycles', 100);

% Predict the responses for 2021 data
CE_OA_Real2021_01 = predict(CEMdl2000_2020, DataVector2021);
[~, CE_ROC_Real2021_01] = predict(CEMdl2000_2020, DataVector2021);
CE_Real2021_OA = sum(CE_OA_Real2021_01 == gtVector2021) / size(gtVector2021, 1);
disp(["CE Real 2021_01 OA= ", num2str(CE_Real2021_OA)])
rocCE_Real2021_01 = rocmetrics(gtVector2021, CE_ROC_Real2021_01, CEMdl2000_2020.ClassNames);
rocCE_Real2021_01.AUC
figure % Plot ROC
tiledlayout(1,1,TileSpacing="loose")
nexttile
plot(rocCE_Real2021_01)
title("ROC CE For Real2021 01 samples")


function obj = myObjectiveFunction(x, data, labels)
    % Train a Bagged Tree Ensemble
    Mdl = fitrensemble(data, labels, 'Method', 'Bag', 'NumLearningCycles', x.numTrees);
   
    % Obtain out-of-bag predictions
    oobPredictions = oobPredict(Mdl);
   
    % Calculate classification error manually
    classificationError = sum(oobPredictions ~= labels) / numel(labels);
   
    % Minimize the classification error
    obj = classificationError;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% KNN OPTIMAZATION  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Choose the optimal k using cross-validation
KNNMdl2000_2020 = fitcknn(DataVector2000_2020,gtVector2000_2020);
cvmodel = crossval(KNNMdl2000_2020, 'KFold', 3); % Perform 3-fold cross-validation
kloss = kfoldLoss(cvmodel,'Mode','individual'); % Get the error rate for each fold
figure % Plot the error rate versus k
plot(cvmodel.ModelParameters.NumNeighbors,kloss,'o-')
xlabel('Number of nearest neighbors')
ylabel('Cross-validated error rate')
title('Choose the optimal k')

% Try different distance metrics
distances = {'euclidean','cityblock','minkowski','mahalanobis'}; % Define a cell array of distance metrics
for i = 1:length(distances) % Loop over the distance metrics
    KNNMdl2000_2020 = fitcknn(DataVector2000_2020,gtVector2000_2020,'Distance',distances{i}); % Fit the model with the distance metric
    [KNN_OA_Real2021_01, ~] = predict(KNNMdl2000_2020,DataVector2021); % Predict the labels for the test data
    KNN_Real2021_OA=sum(KNN_OA_Real2021_01==gtVector2021)/size(gtVector2021,1); % Calculate the overall accuracy
    disp([distances{i},' distance, KNN Real 2021_01 OA= ',num2str(KNN_Real2021_OA)]) % Display the results
end

% Evaluate the performance using different metrics
metrics = classificationMetrics(gtVector2021,KNN_OA_Real2021_01); % Calculate the metrics
disp(metrics) % Display the metrics in a table
figure % Plot the confusion matrix
plotconfusion(gtVector2021,KNN_OA_Real2021_01)
title('Confusion matrix for KNN Real 2021_01')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Preallocating arrays for efficiency
kRange = 2;
kLoss = zeros(length(kRange), 1);

% You need to have the Parallel Computing Toolbox and start a parallel pool
% if one is not already running. If the parallel pool is not started, MATLAB
% will run the parfor loop in serial mode, which will not speed up the process.
% You can start a parallel pool with `parpool` command.

% Use parallel for loop if possible
parfor k = kRange
    % Fit the KNN model with the current k (avoid inside parfor if not supported)
    KNNMdl = fitcknn(DataVector2000_2020, gtVector2000_2020, 'NumNeighbors', k);
    % Perform cross-validation
    cvmodel = crossval(KNNMdl, 'KFold', 3);
    % Compute the average loss for the current k
    kLoss(k) = kfoldLoss(cvmodel);
end
 KNNMdl = fitcknn(DataVector2000_2020, gtVector2000_2020, 'NumNeighbors', 2);
% Plotting is usually not the bottleneck and can't be sped up significantly
% Plot the error rate versus k
figure;
plot(kRange, kLoss, 'o-');
xlabel('Number of nearest neighbors');
ylabel('Cross-validated error rate');
title('Choose the optimal k');

% Assuming KNNMdl is the trained model and DataVector2021 is the new data
% for 2021

% Predict the labels for the 2021 dataset using the trained KNN model
[predictedLabels, scores] = predict(KNNMdl, DataVector2021);


% Calculate the number of unique classes
uniqueClasses = unique(gtVector2021);
numClasses = length(uniqueClasses);

% Preallocate cell arrays for the ROC curve data
[X, Y, T, AUC] = deal(cell(numClasses, 1));

% Calculate ROC for each class using a parallel loop
parfor classIdx = 1:numClasses
    % Construct a logical vector where the current class is 1 and all others are 0
    binaryClass = gtVector2021 == uniqueClasses(classIdx);
    % Calculate the ROC curve for the current class
    [X{classIdx}, Y{classIdx}, T{classIdx}, AUC{classIdx}] = perfcurve(binaryClass, scores(:, classIdx), true);
end




% Calculate confusion matrix with the predicted labels and the true labels (gtVector2021)
[C, order] = confusionmat(gtVector2021, predictedLabels);

% Performance metrics calculations
accuracy = sum(diag(C)) / sum(C(:));
precision = diag(C) ./ sum(C, 2);
recall = diag(C) ./ sum(C, 1)';
F1_scores = 2 * (precision .* recall) ./ (precision + recall);

% Display the metrics
disp(['Accuracy: ', num2str(accuracy)]);
% ... rest of the code for displaying precision, recall, and F1 Score

% Plot confusion matrix
figure;
confusionchart(C, order);
title('Confusion matrix for KNN Model 2021');

% Preallocate cell arrays for the ROC curve data
[X, Y, T, AUC] = deal(cell(numClasses, 1));

% Calculate ROC for each class using a parallel loop
parfor classIdx = 1:numClasses
    [X{classIdx}, Y{classIdx}, T{classIdx}, AUC{classIdx}] = perfcurve(gtVector2021 == classIdx, scores(:, classIdx), classIdx);
end

% Plot ROC curves - plotting cannot be optimized
% Preallocate the legendEntry cell array
legendEntry = cell(numClasses, 1);

% Plot ROC curves for each class
figure;
hold on;
legendEntry = cell(numClasses, 1); % Initialize the legend entry cell array

for classIdx = 1:numClasses
    % Plot the ROC curve for the current class
    plot(X{classIdx}, Y{classIdx});
    % Create the legend entry for the current class
    legendEntry{classIdx} = ['Class ' num2str(classIdx) ' (AUC: ' num2str(AUC{classIdx}, '%.2f') ')'];
end

% Customize the plot
xlabel('False positive rate');
ylabel('True positive rate');
title('ROC Curves for Each Class');

% Only display legend entries for the plotted lines
legend(legendEntry, 'Location', 'Best');
hold off; % Release the hold on the current figure


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MNB optimization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Assuming you have loaded your data and defined variables

% Fit a Naive Bayes model using cross-validation
cv = cvpartition(size(DataVector2000_2020, 1), 'KFold', 3);
crossValScores = zeros(cv.NumTestSets, 1);

for i = 1:cv.NumTestSets
    trainData = DataVector2000_2020(training(cv, i), :);
    trainLabels = gtVector2000_2020(training(cv, i), :);
    valData = DataVector2000_2020(test(cv, i), :);
    valLabels = gtVector2000_2020(test(cv, i), :);

    MNBMdl = fitcnb(trainData, trainLabels);
    predictions = predict(MNBMdl, valData);
    crossValScores(i) = sum(predictions == valLabels) / length(valLabels);
end

meanAccuracy = mean(crossValScores);
disp(["Mean CV Accuracy: ", num2str(meanAccuracy)]);

% Train final model on all data
finalMdl = fitcnb(DataVector2000_2020, gtVector2000_2020);

% Predict on validation set for final model
[MNB_pred_val, ~] = predict(finalMdl, valData);

% Compute ROC curve
[X,Y,T,AUC] = perfcurve(valLabels, MNB_pred_val, 1); % Assuming positive class is 1

% Display AUC
disp(["AUC: ", num2str(AUC)]);

% Plot ROC curve
figure;
plot(X, Y);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('Receiver Operating Characteristic (ROC) Curve');
grid on;

% Rank features using MRMR
idx = fscmrmr(DataVector2000_2020, gtVector2000_2020);

% Display the ranked feature indices
disp("Ranked feature indices (from most important to least important):");
disp(idx);

%%%%%%%%%%%%%%%%%%%%%%%% SVM%%%%%%%%%%%%%%%%%%%%%%%%
% Assuming you have loaded your data and defined variables

% Fit a support vector machine (SVM) model
svmMdl2000_2020 = fitcsvm(DataVector2000_2020, gtVector2000_2020);

% Predict on validation set for SVM
[SVM_OA_Real2021_01, ~] = predict(svmMdl2000_2020, DataVector2021);

% Calculate overall accuracy for SVM
SVM_Real2021_OA = sum(SVM_OA_Real2021_01 == gtVector2021) / size(gtVector2021, 1);
disp(["SVM Real 2021_01 OA: ", num2str(SVM_Real2021_OA)]);

% Compute ROC curve for SVM
[~, SVM_ROC_Real2021_01] = predict(svmMdl2000_2020, DataVector2021);
rocSVM_Real2021_01 = rocmetrics(gtVector2021, SVM_ROC_Real2021_01, svmMdl2000_2020.ClassNames);
disp(["AUC for SVM Real 2021_01: ", num2str(rocSVM_Real2021_01.AUC)]);

% Plot ROC curve
figure;
plot(rocSVM_Real2021_01);
title("ROC SVM For Real2021 01 samples");
xlabel('False Positive Rate');
ylabel('True Positive Rate');
grid on;
%%%%%%%%%%%%%%%%%%%%%%%svm and feature importance%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check if the SVM model is linear
if strcmp(svmMdl2000_2020.KernelFunction, 'linear')
    % Get the absolute values of the weights
    weights = abs(svmMdl2000_2020.Beta);
    % Rank features based on the weights
    [sortedWeights, featureIdx] = sort(weights, 'descend');
    
    % Display ranked features
    disp('Ranked feature indices (from most important to least important):');
    disp(featureIdx);
else
    disp('SVM model is not linear, consider alternative methods for feature importance.');
end
numFeatures = size(DataVector2000_220, 2);
importanceScores = zeros(numFeatures, 1);

for i = 1:numFeatures
    % Permute feature i
    tempData = DataVector2000_2020;
    tempData(:, i) = tempData(randperm(size(tempData, 1)), i);
    
    % Predict with permuted data
    [~, scorePermuted] = predict(svmMdl2000_2020, tempData);
    
    % Calculate accuracy with permuted feature
    accuracyPermuted = sum(scorePermuted == gtVector2000_2020) / length(gtVector2000_2020);
    
    % Compute importance as the decrease in accuracy
    importanceScores(i) = SVM_Real2021_OA - accuracyPermuted;
end

% Rank features based on the importance scores
[~, featureIdxPerm] = sort(importanceScores, 'descend');

% Display ranked features based on permutation importance
disp('Ranked feature indices based on permutation importance (from most important to least important):');
disp(featureIdxPerm);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% demster shaffer %%%%%%%%%%%%%%%%%%%%%%%

% Step 1: Prepare the Dempster-Shafer Model
% Define mass functions for each model's predictions
% For simplicity, we assume equal belief distribution among the models
mass_RF = 0.25 * (CE_OA_Real2021_01 == gtVector2021);
mass_KNN = 0.25 * (KNN_OA_Real2021_01 == gtVector2021);
mass_MNB = 0.25 * (MNB_OA_Real2021_01 == gtVector2021);
mass_SVM = 0.25 * (SVM_OA_Real20121_01 == gtVector2021);

% Step 2: Calculate the Combined Beliefs
% Combine the beliefs using Dempster's rule of combination
combined_beliefs = dempster_rule(mass_RF, mass_KNN, mass_MNB, mass_SVM);

% Step 3: Generate the Combined Predictions
% Use the combined beliefs to generate the final predictions
combined_predictions = combined_beliefs > 0.5; % Simple thresholding for binary classification

% Step 4: Calculate AUC and ROC Accuracy
% Use the rocmetrics function to calculate the ROC curve and AUC
% Now use 'classNames' in rocmetrics or wherever it's needed
[Combined_ROC, ~, ~, AUC] = rocmetrics(gtVector2021, combined_predictions, classNames);


% Display the AUC
disp(["AUC for Combined Model: ", num2str(AUC)]);

% Plot the ROC curve
figure;
plot(Combined_ROC);
title("ROC for Combined Model");

% Ensure no script code follows the local function definitions

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% demster shaffer2 %%%%%%%%%%%%%%%%%%%%

% Step 1: Prepare the Dempster-Shafer Model
% Define mass functions for each model's predictions
% For simplicity, we assume equal belief distribution among the models
mass_RF = 0.25 * (CE_OA_Real2021_01 == gtVector2021);
mass_KNN = 0.25 * (KNN_OA_Real2021_01 == gtVector2021);
mass_MNB = 0.25 * (MNB_OA_Real2021_01 == gtVector2021);
mass_SVM = 0.25 * (SVM_OA_Real20121_01 == gtVector2021);

% Check for NaN values and remove corresponding samples
validIndices = ~isnan(mass_RF) & ~isnan(mass_KNN) & ~isnan(mass_MNB) & ~isnan(mass_SVM);
mass_RF = mass_RF(validIndices);
mass_KNN = mass_KNN(validIndices);
mass_MNB = mass_MNB(validIndices);
mass_SVM = mass_SVM(validIndices);
gtVector2021 = gtVector2021(validIndices);

% Step 2: Calculate the Combined Beliefs
% Combine the beliefs using Dempster's rule of combination
combined_beliefs = dempster_rule(mass_RF, mass_KNN, mass_MNB, mass_SVM);

% Step 3: Generate the Combined Predictions
% Use the combined beliefs to generate the final predictions
combined_predictions = combined_beliefs > 0.5; % Simple thresholding for binary classification

% Ensure gtVector2021 is a categorical vector
if ~iscategorical(gtVector2021)
    % Convert gtVector2021 to categorical if it is not already
    gtVector2021 = categorical(gtVector2021);
end

% Step 4: Calculate AUC and ROC Accuracy
% Define class names and positive class
classNames = categories(gtVector2021); % Get unique class names from categorical vector
positiveClass = classNames{1}; % Assume the first class is the positive class

% Convert categorical gtVector2021 to numeric logical array for positive class
binaryLabels = gtVector2021 == positiveClass;

% Ensure combined_beliefs is a column vector
if size(combined_beliefs, 2) ~= 1
    combined_beliefs = combined_beliefs(:);
end

% Display sizes of combined_beliefs and binaryLabels
disp(['Size of combined_beliefs: ', num2str(size(combined_beliefs))]);
disp(['Size of binaryLabels: ', num2str(size(binaryLabels))]);

% Check if the sizes match
if length(combined_beliefs) ~= length(binaryLabels)
    error('The number of elements in combined_beliefs and binaryLabels must be the same.');
end

% Calculate ROC and AUC using perfcurve
try
    [X, Y, ~, AUC] = perfcurve(binaryLabels, combined_beliefs, true); % 'true' for positive class
    
    % Display AUC
    disp(['AUC for Combined Model: ', num2str(AUC)]);

    % Plot ROC curve
    figure;
    plot(X, Y);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title('ROC Curve for Combined Model');
catch ME
    disp('An error occurred while calculating ROC and AUC:');
    disp(ME.message);
end

% Define the Dempster's rule function as a local function
function combined_beliefs = dempster_rule(mass_RF, mass_KNN, mass_MNB, mass_SVM)
    % Normalize the mass functions to ensure they sum to 1
    total_mass = mass_RF + mass_KNN + mass_MNB + mass_SVM;
    total_mass(total_mass == 0) = 1; % Avoid division by zero
    mass_RF = mass_RF ./ total_mass;
    mass_KNN = mass_KNN ./ total_mass;
    mass_MNB = mass_MNB ./ total_mass;
    mass_SVM = mass_SVM ./ total_mass;

    % Calculate the combined mass function
    combined_beliefs = mass_RF .* mass_KNN .* mass_MNB .* mass_SVM;
end



























