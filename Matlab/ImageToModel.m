% Load the stereoParameters object.
load('handshakeStereoParams.mat');

% Visualize camera extrinsics.
% showExtrinsics(stereoParams);

frame = 1;

% Create video file readers and the Video Player
videoFileLeft = 'images/left_low_light.avi';
videoFileRight = 'images/right_low_light.avi';

readerLeft = VideoReader(videoFileLeft);
readerRight = VideoReader(videoFileRight);
player = vision.VideoPlayer('Position', [20,200,740 560]);

while(hasFrame(readerLeft) && hasFrame(readerRight))
    % Read and Rectify Video Frames
    frameLeft = readFrame(readerLeft);
    frameRight = readFrame(readerRight);
    frame = frame + 1;
    if (frame == 3)
        [frameLeftRect, frameRightRect] = ...
            rectifyStereoImages(frameLeft, frameRight, stereoParams);
        
        figure;
        imshow(stereoAnaglyph(frameLeftRect, frameRightRect));
        title('Rectified Video Frames');
        
        % Compute Disparity
        frameLeftGray  = rgb2gray(frameLeftRect);
        frameRightGray = rgb2gray(frameRightRect);
            
        disparityMap = disparitySGM(frameLeftGray, frameRightGray);
        figure;
        imshow(disparityMap, [0, 64]);
        title('Disparity Map');
        colormap jet
        colorbar
        
        % Reconstruct the 3-D Scene
        points3D = reconstructScene(disparityMap, stereoParams);
        
        % Convert to meters and create a pointCloud object
        points3D = points3D ./ 1000;
        ptCloud = pointCloud(points3D, 'Color', frameLeftRect);
        
        % Create a streaming point cloud viewer
        player3D = pcplayer([-5, 5], [-5, 5], [0, 20], 'VerticalAxis', 'y', ...
            'VerticalAxisDir', 'down');
        
        % Visualize the point cloud
        view(player3D, ptCloud);
    end
end

% Clean up
release(player);