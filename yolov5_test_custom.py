from PIL import Image
import torch
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

#argparser with default values
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp', type=str, default='exp', help='experiment number')
    return parser.parse_args()

exp = f'{argparser().exp}'
predition_plots_dir = f'runs/train/{exp}/predition_plots/'
print(f'predition_plots_dir = {predition_plots_dir}')

if not os.path.exists(predition_plots_dir):
    os.mkdir(predition_plots_dir)
    print(f"Directory '{predition_plots_dir}' created.")
else:
    print(f"Directory '{predition_plots_dir}' already exists.")

def plot_results(results, img, cross, i, label):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    x,y = np.load(cross)
    #read the label file
    label = np.loadtxt(label)*500
    ax.scatter(x, y, c='b', s=0.1, label=f'truth cross\ncenter=[{label[1]}, {label[2]}]')
    x_min = float(results['xmin'])
    y_min = float(results['ymin'])
    x_max = float(results['xmax'])
    y_max = float(results['ymax'])

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    print(f'x = {x_center}, y = {y_center}')
    #plot a rectangle with the prerecet = dicted x_min, y_min, x_max, y_max
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, color='r', fill=False, label = f'predicted box\ncenter=[{round(x_center,1), round(y_center,1)}]')
    #rect = plt.Rectangle((x_center-width/2, y_center-height/2), width, height, fill=False, color='r')
    ax.add_patch(rect)
    #ax.text(0.5, 0.5, f"Precision: {float(results['confidence'])}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r')
    #print as a title of the plot the rms error between the predicted center and the center of the cross
    ax.set_title(f"center RMS error: {np.round(np.sqrt((x_center-label[1])**2+(y_center-label[2])**2),3)} [pixels]")
    plt.legend(loc='best')
    #ax.text(0.5, 0.4, f'RMS error: {np.sqrt((x_center-x)**2+(y_center-y)**2)}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r')
    plt.savefig(f'{predition_plots_dir}results{i}.png')
    return np.sqrt((x_center-label[1])**2+(y_center-label[2])**2), float(results['confidence'])

def plot_results2(results, img, i, label):

    print(f"predition: class = {str(results['name'])}, x_min = {round(float(results['xmin']),3)}, y_min = {round(float(results['ymin']),3)}, x_max = {round(float(results['xmax']),3)}, y_max = {round(float(results['ymax']),3)}, confidence = {round(float(results['confidence']),3)}")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    #read the label file
    label = np.loadtxt(label)*500
    print(f"truth:     class = {'cross' if label[0]==0 else 'flexPad'}, x = {label[1]}, y = {label[2]}")
    x_min = float(results['xmin'])
    y_min = float(results['ymin'])
    x_max = float(results['xmax'])
    y_max = float(results['ymax'])
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    # print(f"predition:  class = {str(results['name'])}, x = {x_center}, y = {y_center}, confidence = {float(results['confidence'])}")
    # print(f"truth:      class = {'cross' if label[0]==0 else 'flexPad'}, x = {label[1]}, y = {label[2]}")

    #plot a rectangle with the prerecet = dicted x_min, y_min, x_max, y_max
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, color='r', fill=False, label = f'predicted box\ncenter=[{round(x_center,1), round(y_center,1)}]')
    #rect = plt.Rectangle((x_center-width/2, y_center-height/2), width, height, fill=False, color='r')
    ax.add_patch(rect)
    #ax.text(0.5, 0.5, f"Precision: {float(results['confidence'])}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r')
    #print as a title of the plot the rms error between the predicted center and the center of the cross
    ax.set_title(f"center RMS error: {np.round(np.sqrt((x_center-label[1])**2+(y_center-label[2])**2),3)} [pixels]")
    plt.legend(loc='best')
    #ax.text(0.5, 0.4, f'RMS error: {np.sqrt((x_center-x)**2+(y_center-y)**2)}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r')
    plt.savefig(f'{predition_plots_dir}results{i}.png')
    return np.sqrt((x_center-label[1])**2+(y_center-label[2])**2), float(results['confidence'])

def plot_results_circ(results, img, i, label):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    #read the label file
    print(label)
    label = np.loadtxt(label)*500

    x_min = float(results['xmin'])
    y_min = float(results['ymin'])
    x_max = float(results['xmax'])
    y_max = float(results['ymax'])

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    print(f'x = {x_center}, y = {y_center}')
    #plot a rectangle with the prerecet = dicted x_min, y_min, x_max, y_max
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, color='r', fill=False, label = f'predicted box\ncenter=[{round(x_center,1), round(y_center,1)}]')
    #rect = plt.Rectangle((x_center-width/2, y_center-height/2), width, height, fill=False, color='r')
    ax.add_patch(rect)
    #ax.text(0.5, 0.5, f"Precision: {float(results['confidence'])}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r')
    #print as a title of the plot the rms error between the predicted center and the center of the cross
    ax.set_title(f"center RMS error: {np.round(np.sqrt((x_center-label[1])**2+(y_center-label[2])**2),3)} [pixels]")
    plt.legend(loc='best')
    #ax.text(0.5, 0.4, f'RMS error: {np.sqrt((x_center-x)**2+(y_center-y)**2)}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r')
    plt.savefig(f'{predition_plots_dir}results{i}.png')
    return np.sqrt((x_center-label[1])**2+(y_center-label[2])**2), float(results['confidence'])

#get predictions on test images
#images_path = '../datasets/images/test'
images_path = '../datasets/flex_circ/test/images'
images = glob.glob(os.path.join(images_path, '*.jpg'))
#the original one and the only tested
#crosses = [os.path.join(crosses_path, os.path.basename(x).split('.')[0]+'.npy') for x in images]
#print(crosses)
print(f'images:\n{images}')
#Model
rmses = []
confidence = []
model = torch.hub.load('.', 'custom', path=f'runs/train/{exp}/weights/best.pt', source='local')  # local repo
for i in range(len(images)):
    print(images[i].split('/')[-1])
    img = Image.open(images[i])
    results = model(img, size=500)  # run once
    results = results.pandas().xyxy[0]
    #get labels with the same name of the image but changing the extension
    #labels = glob.glob(os.path.join(images_path.split('images')[0]+'labels/test', os.path.basename(images[i]).split('.')[0]+'.txt'))
    #labels = glob.glob(os.path.join(images_path.split('flex_circle')[0]+'labels', os.path.splitext(images[i])[0]+'.txt'))
    labels =  os.path.splitext(images[i])[0].replace("images", "labels") + ".txt"
    print(f'labels:\n{labels}')
    # with open(labels[0]) as f:
    #     label = f.readline().strip().split(' ')
    # label = [float(x) for x in label]
    # print(f'label: {label}')
    print(f'results: {results}')
    #consider only the line in results that has the major confidence
    results = results[results['confidence']==results['confidence'].max()]
    print(f'results: {results}')

    #rmse, conf = plot_results(results, img, crosses[i], images[i].split('/')[-1].split('.')[0], labels[i])
    rmse, conf = plot_results_circ(results, img, images[i].split('/')[-1].split('.')[0], labels)
    #rmse, conf = plot_results2(results, img, images[i].split('/')[-1].split('.')[0], labels[0])
    rmses.append(rmse)
    confidence.append(conf)
    

#plot the rmse histogram
plt.figure()
plt.hist(rmses)
plt.xlabel('RMS error [pixels]')
plt.ylabel('Counts')
plt.savefig(f'{predition_plots_dir}rmse.png')

#plot confidence vs rmse and fit it with a line
plt.figure()
plt.scatter(confidence, rmses)
plt.xlabel('Confidence')
plt.ylabel('RMS error [pixels]')
#fit a line
m, b = np.polyfit(confidence, rmses, 1)
plt.plot(confidence, m*np.array(confidence) + b, color='r', label=f'fit: y={round(m,3)}x+{round(b,3)}')
plt.legend(loc='best')
plt.savefig(f'{predition_plots_dir}confidence_vs_rmse.png')
