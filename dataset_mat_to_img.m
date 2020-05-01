% original script credits to Akira Agata at:
% https://uk.mathworks.com/matlabcentral/answers/445766-how-to-save-images-as-jpgs-from-mat-files-struct

% change inputFolder and outputFolder if necessary
% so that they point to the directory containing .mat files
inputFolder = 'D:\where\dataset\is\stored';
outputFolder = pwd; % Please change, if needed.
fileList = dir(fullfile(inputFolder,'*.mat'));

for kk = 1:numel(fileList)
  S = load(fullfile(pwd,fileList(kk).name));
  I = S.cjdata.image;
  I = mat2gray(I);
  fileName = strrep(fileList(kk).name,'.mat','.tiff');
  imwrite(I,fullfile(outputFolder,fileName));
end
