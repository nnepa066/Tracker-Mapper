% Load the stereoParameters object.
load('webcamsSceneReconstruction.mat');

I1 = imread('rectified_left.png');
I2 = imread('rectified_right.png');

[J1, J2] = rectifyStereoImages(I1,I2,stereoParams);

figure 
imshow(cat(3,J1(:,:,1),J2(:,:,2:3)),'InitialMagnification',50);

disparityMap = disparitySGM(im2gray(J1),im2gray(J2));
figure 
imshow(disparityMap,[0,64],'InitialMagnification',50);

xyzPoints = reconstructScene(disparityMap,stereoParams);

Z = xyzPoints(:,:,3);
mask = repmat(Z > 3200 & Z < 3700,[1,1,3]);
J1(~mask) = 0;
imshow(J1,'InitialMagnification',50);

% Clean up
release(player);
