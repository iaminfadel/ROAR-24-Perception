downloadURL  = "https://github.com/AprilRobotics/apriltag-imgs/archive/master.zip";
dataFolder   = fullfile(tempdir,"apriltag-imgs",filesep); 
options      = weboptions('Timeout', Inf);
zipFileName  = fullfile(dataFolder,"apriltag-imgs-master.zip");
folderExists = exist(dataFolder,"dir");

% Create a folder in a temporary directory to save the downloaded file.
if ~folderExists  
    mkdir(dataFolder); 
    disp("Downloading apriltag-imgs-master.zip (60.1 MB)...") 
    websave(zipFileName,downloadURL,options); 
    
    % Extract contents of the downloaded file.
    disp("Extracting apriltag-imgs-master.zip...") 
    unzip(zipFileName,dataFolder); 
end