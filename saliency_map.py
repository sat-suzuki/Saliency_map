import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import scipy.spatial.distance as dis
import sys
import caffe
import os

caffe_root = '/usr/local/DL-Box/digits-2.0/caffe/'
sys.path.insert(0, caffe_root + 'python')
caffe.set_mode_gpu()

def get_crop_image_conv3(net, no_feature_map):
    inputs = net.blobs['conv3'].data[4,no_feature_map:no_feature_map+1,:,:]
    temp = np.squeeze(net.blobs['conv3'].data[4,no_feature_map:no_feature_map+1,:,:])
    #print 'conv5 blob shape:', temp.shape
    index = np.argmax(temp)
    x = index/temp.shape[0]
    y = index % temp.shape[0]
    print 'max index:',(x,y),np.max(temp), temp[x,y]

    temp_shape = net.blobs['pool2'].data.shape
    boundary = [max(x-1,0), min(x+1, temp_shape[2]-1), max(y-1,0), min(y+1, temp_shape[3]-1)]
    
    temp_shape = net.blobs['cccp2'].data.shape
    boundary1 = [boundary[0]*2, min((boundary[1])*2+2, temp_shape[2]-1), boundary[2]*2, min(boundary[3]*2+2, temp_shape[3]-1)]
    
    temp_shape = net.blobs['pool1'].data.shape
    boundary2 = [max(boundary1[0]-2,0), min(boundary1[1]+2, temp_shape[2]-1), max(boundary1[2]-2,0), min(boundary1[3]+2, temp_shape[3]-1)]
    
    temp_shape = net.blobs['cccp1'].data.shape
    boundary3 = [boundary2[0]*2, min((boundary2[1])*2+2, temp_shape[2]-1), boundary2[2]*2, min(boundary2[3]*2+2, temp_shape[3]-1)]

    max_idx = [x,y]
    return boundary3, max_idx

def make_saliency_map_conv3(net, no_feature_map, img_name, occ_boundary, max_idx):
    image = caffe.io.load_image(img_name)
    occ_boundary[0] = max(0, boundary[0]*4);
    occ_boundary[1]= occ_boundary[1]*4
    occ_boundary[2] = max(0,boundary[2]*4);
    occ_boundary[3] = occ_boundary[3]*4
    sal_size_seven = [int((occ_boundary[1]-occ_boundary[0]-7)/4)+1,int((occ_boundary[3]-occ_boundary[2]-7)/4)+1]
    sal_save_seven = np.zeros([sal_size_seven[0],sal_size_seven[1]])
    # occ -> 7x7
    for j in range(sal_size_seven[1]):
        for i in range(sal_size_seven[0]):
            print i
            transformed_image = transformer.preprocess('data', image)   
            occ_reg = np.zeros([1,4])
            occ_reg = [max(0,boundary[0]+i*4),min(boundary[0]+i*4+7,boundary[1]),max(0,boundary[2]+j*4),min(boundary[2]+j*4+7,boundary[3])]
            transformed_image[:,occ_reg[0]:occ_reg[1],occ_reg[2]:occ_reg[3]] = np.mean(transformed_image)
            net.blobs['data'].data[...] = transformed_image
            output = net.forward()
            temp = np.squeeze(net.blobs['conv3'].data[4,no_feature_map:no_feature_map+1,:,:])
            sal_save_seven[i,j] = temp[max_idx[0],max_idx[1]]

    sal_size_thi = [int((occ_boundary[1]-occ_boundary[0]-7)/4)+1,int((occ_boundary[3]-occ_boundary[2]-7)/4)+1]
    sal_save_thi = np.zeros([sal_size[0],sal_size[1]])
    # occ -> 13x13
    return sal_save_seven

def get_crop_image_cccp3(net, no_feature_map):
    inputs = net.blobs['cccp3'].data[4,no_feature_map:no_feature_map+1,:,:]
    temp = np.squeeze(net.blobs['cccp3'].data[4,no_feature_map:no_feature_map+1,:,:])
    #print 'conv5 blob shape:', temp.shape
    index = np.argmax(temp)
    x = index/temp.shape[0]
    y = index % temp.shape[0]
    print 'max index:',(x,y),np.max(temp), temp[x,y]

    temp_shape = net.blobs['pool2'].data.shape
    boundary = [max(x-1,0), min(x+1, temp_shape[2]-1), max(y-1,0), min(y+1, temp_shape[3]-1)]
    
    temp_shape = net.blobs['cccp2'].data.shape
    boundary1 = [boundary[0]*2, min((boundary[1])*2+2, temp_shape[2]-1), boundary[2]*2, min(boundary[3]*2+2, temp_shape[3]-1)]
    
    temp_shape = net.blobs['pool1'].data.shape
    boundary2 = [max(boundary1[0]-2,0), min(boundary1[1]+2, temp_shape[2]-1), max(boundary1[2]-2,0), min(boundary1[3]+2, temp_shape[3]-1)]
    
    temp_shape = net.blobs['cccp1'].data.shape
    boundary3 = [boundary2[0]*2, min((boundary2[1])*2+2, temp_shape[2]-1), boundary2[2]*2, min(boundary2[3]*2+2, temp_shape[3]-1)]

    max_idx = [x,y]
    return boundary3, max_idx


def make_saliency_map_cccp3(net, no_feature_map, img_name, occ_boundary,max_idx):
    image = caffe.io.load_image(img_name)
    occ_boundary[0] = max(0, boundary[0]*4);
    occ_boundary[1]= occ_boundary[1]*4
    occ_boundary[2] = max(0,boundary[2]*4);
    occ_boundary[3] = occ_boundary[3]*4
    sal_size_seven = [int((occ_boundary[1]-occ_boundary[0]-7)/4)+1,int((occ_boundary[3]-occ_boundary[2]-7)/4)+1]
    sal_save_seven = np.zeros([sal_size_seven[0],sal_size_seven[1]])
    # occ -> 7x7
    for j in range(sal_size_seven[1]):
        for i in range(sal_size_seven[0]):
            print i
            transformed_image = transformer.preprocess('data', image)   
            occ_reg = np.zeros([1,4])
            occ_reg = [max(0,boundary[0]+i*4),min(boundary[0]+i*4+7,boundary[1]),max(0,boundary[2]+j*4),min(boundary[2]+j*4+7,boundary[3])]
            transformed_image[:,occ_reg[0]:occ_reg[1],occ_reg[2]:occ_reg[3]] = np.mean(transformed_image)
            net.blobs['data'].data[...] = transformed_image
            output = net.forward()
            temp = np.squeeze(net.blobs['cccp3'].data[4,no_feature_map:no_feature_map+1,:,:])
            sal_save_seven[i,j] = temp[max_idx[0],max_idx[1]]

    #sal_size_thi = [int((occ_boundary[1]-occ_boundary[0]-7)/4)+1,int((occ_boundary[3]-occ_boundary[2]-7)/4)+1]
    #sal_save_thi = np.zeros([sal_size_thi[0],sal_size_thi[1]])
    return sal_save_seven



def get_crop_image_conv2(net, no_feature_map):
    inputs = net.blobs['conv2'].data[4,no_feature_map:no_feature_map+1,:,:]
    temp = np.squeeze(net.blobs['conv2'].data[4,no_feature_map:no_feature_map+1,:,:])
    #print 'conv5 blob shape:', temp.shape
    index = np.argmax(temp)
    x = index/temp.shape[0]
    y = index % temp.shape[0]
    print 'max index:',(x,y),np.max(temp), temp[x,y]
    
    temp_shape = net.blobs['pool1'].data.shape
    boundary2 = [max(x-2,0), min(x+2, temp_shape[2]-1), max(y-2,0), min(y+2, temp_shape[3]-1)]
    
    temp_shape = net.blobs['cccp1'].data.shape
    boundary3 = [boundary2[0]*2, min((boundary2[1])*2+2, temp_shape[2]-1), boundary2[2]*2, min(boundary2[3]*2+2, temp_shape[3]-1)]

    max_idx = [x,y]
    return boundary3, max_idx

def make_saliency_map_conv2(net, no_feature_map, img_name, occ_boundary, max_idx):
    image = caffe.io.load_image(img_name)
    occ_boundary[0] = max(0, boundary[0]*4);
    occ_boundary[1]= occ_boundary[1]*4
    occ_boundary[2] = max(0,boundary[2]*4);
    occ_boundary[3] = occ_boundary[3]*4
    sal_size_seven = [int((occ_boundary[1]-occ_boundary[0]-7)/4)+1,int((occ_boundary[3]-occ_boundary[2]-7)/4)+1]
    sal_save_seven = np.zeros([sal_size_seven[0],sal_size_seven[1]])
    # occ -> 7x7
    for j in range(sal_size_seven[1]):
        for i in range(sal_size_seven[0]):
            print i
            transformed_image = transformer.preprocess('data', image)   
            occ_reg = np.zeros([1,4])
            occ_reg = [max(0,boundary[0]+i*4),min(boundary[0]+i*4+7,boundary[1]),max(0,boundary[2]+j*4),min(boundary[2]+j*4+7,boundary[3])]
            transformed_image[:,occ_reg[0]:occ_reg[1],occ_reg[2]:occ_reg[3]] = np.mean(transformed_image)
            net.blobs['data'].data[...] = transformed_image
            output = net.forward()
            temp = np.squeeze(net.blobs['conv2'].data[4,no_feature_map:no_feature_map+1,:,:])
            sal_save_seven[i,j] = temp[max_idx[0],max_idx[1]]

    #sal_size_thi = [int((occ_boundary[1]-occ_boundary[0]-7)/4)+1,int((occ_boundary[3]-occ_boundary[2]-7)/4)+1]
    #sal_save_thi = np.zeros([sal_size_thi[0],sal_size_thi[1]])
    # occ -> 13x13
    return sal_save_seven


#net = caffe.Net('deploy.prototxt','nin_imagenet_train_iter_450000.caffemodel',caffe.TEST)
net = caffe.Net('deploy.prototxt','caffenet_nin_train_iter_450000.caffemodel',caffe.TEST)
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
net.blobs['data'].reshape(50, 3, 227, 227)

##################################################################################################################
#mode: 1->only saliency map, 2->with crop image
mode = 2

#please select layer number
layer = 2

#cccp layer?: 0->No, 1->Yes
cccp = 0
#################################################################################################################


command = mode*100 + layer*10 + cccp

if command == 130:
    for k in range(384):
        z = np.zeros(9)
        idx = np.zeros(9)
        f = open('img_name_imageNet_test.txt')
        data1 = f.read()
        f.close()
        lines1 = data1.split('\n')
        for i in range(len(lines1)-1):
            image = caffe.io.load_image(lines1[i])
            transformed_image = transformer.preprocess('data', image)
            # copy the image data into the memory allocated for the net
            net.blobs['data'].data[...] = transformed_image
            ### perform classification
            output = net.forward()
            hoge = net.blobs['conv3'].data[4,:,:,:]
            if np.max(hoge[k,:,:]) > np.min(z):
                idx[np.argmin(z)] = i
                z[np.argmin(z)] = np.max(hoge[k,:,:])

        no_feature_map = k
        print 'map No:', no_feature_map
    
        for j in range(9):
            image = caffe.io.load_image(lines1[int(idx[j])])
            transformed_image = transformer.preprocess('data', image)
            # copy the image data into the memory allocated for the net
            net.blobs['data'].data[...] = transformed_image
            ### perform classification
            output = net.forward()
            boundary, max_idx = get_crop_image_conv3(net, no_feature_map)
            sali_map = make_saliency_map_conv3(net,no_feature_map, lines1[int(idx[j])],boundary,max_idx)
            #xmin = max(0, boundary[0]*4);
            #xmax = boundary[1]*4
            #ymin = max(0,boundary[2]*4);
            #ymax = boundary[3]*4
            #img = np.transpose(net.blobs['data'].data[4,:,:,:],(1,2,0))
            #img = (img-np.min(img))/(np.max(img)-np.min(img))
            #img = img[:,:,::-1]
            plt.subplot(3,3,j)
            plt.imshow(sali_map);plt.axis('off')
        
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
        plt.savefig('vis_result/conv3_slc_map/saliency_map_of_No'+ str(k) + '.png')

elif command == 230:
    for k in range(384):
        z = np.zeros(9)
        idx = np.zeros(9)
        f = open('img_name_imageNet_test.txt')
        data1 = f.read()
        f.close()
        lines1 = data1.split('\n')
        for i in range(len(lines1)-1):
            image = caffe.io.load_image(lines1[i])
            transformed_image = transformer.preprocess('data', image)
            # copy the image data into the memory allocated for the net
            net.blobs['data'].data[...] = transformed_image
            ### perform classification
            output = net.forward()
            hoge = net.blobs['conv3'].data[4,:,:,:]
            if np.max(hoge[k,:,:]) > np.min(z):
                idx[np.argmin(z)] = i
                z[np.argmin(z)] = np.max(hoge[k,:,:])

            no_feature_map = k
            print 'map No:', no_feature_map
    
        for j in range(9):
            image = caffe.io.load_image(lines1[int(idx[j])])
            transformed_image = transformer.preprocess('data', image)
            # copy the image data into the memory allocated for the net
            net.blobs['data'].data[...] = transformed_image
            ### perform classification
            output = net.forward()
            boundary, max_idx = get_crop_image_conv3(net, no_feature_map)
            sali_map = make_saliency_map_conv3(net,no_feature_map, lines1[int(idx[j])],boundary,max_idx)
            #xmin = max(0, boundary[0]*4);
            #xmax = boundary[1]*4
            #ymin = max(0,boundary[2]*4);
            #ymax = boundary[3]*4
            #img = np.transpose(net.blobs['data'].data[4,:,:,:],(1,2,0))
            #img = (img-np.min(img))/(np.max(img)-np.min(img))
            #img = img[:,:,::-1]
            plt.subplot(3,3,j)
            plt.imshow(sali_map);plt.axis('off')

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
        plt.savefig('vis_result/conv3_slc_map/saliency_map_of_No'+ str(k) + '.png')

        for j in range(9):
            image = caffe.io.load_image(lines1[int(idx[j])])
            transformed_image = transformer.preprocess('data', image)
            # copy the image data into the memory allocated for the net
            net.blobs['data'].data[...] = transformed_image
            ### perform classification
            output = net.forward()
            boundary, max_idx = get_crop_image_conv3(net, no_feature_map)
            xmin = max(0, boundary[0]*4);
            xmax = boundary[1]*4
            ymin = max(0,boundary[2]*4);
            ymax = boundary[3]*4
            img = np.transpose(net.blobs['data'].data[4,:,:,:],(1,2,0))
            img = (img-np.min(img))/(np.max(img)-np.min(img))
            img = img[:,:,::-1]
            plt.subplot(3,3,j)
            plt.imshow(img[xmin:xmax,ymin:ymax,:]);plt.axis('off')
            
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
        plt.savefig('vis_result/conv3_slc_map/crop_img_of_No'+ str(k) + '.png')

elif command == 131:
    for k in range(384):
        z = np.zeros(9)
        idx = np.zeros(9)
        f = open('img_name_imageNet_test.txt')
        data1 = f.read()
        f.close()
        lines1 = data1.split('\n')
        for i in range(len(lines1)-1):
            image = caffe.io.load_image(lines1[i])
            transformed_image = transformer.preprocess('data', image)
            # copy the image data into the memory allocated for the net
            net.blobs['data'].data[...] = transformed_image
            ### perform classification
            output = net.forward()
            hoge = net.blobs['cccp3'].data[4,:,:,:]
            if np.max(hoge[k,:,:]) > np.min(z):
                idx[np.argmin(z)] = i
                z[np.argmin(z)] = np.max(hoge[k,:,:])

        no_feature_map = k
        print 'map No:', no_feature_map
        
        for j in range(9):
            image = caffe.io.load_image(lines1[int(idx[j])])
            transformed_image = transformer.preprocess('data', image)
            # copy the image data into the memory allocated for the net
            net.blobs['data'].data[...] = transformed_image
            ### perform classification
            output = net.forward()
            boundary, max_idx = get_crop_image_cccp3(net, no_feature_map)
            sali_map = make_saliency_map_cccp3(net,no_feature_map, lines1[int(idx[j])],boundary,max_idx)
            plt.subplot(3,3,j)
            #plt.imshow(sali_map,aspect='auto');plt.axis('off')
            plt.imshow(sali_map);plt.axis('off')
            
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.05)
        plt.savefig('vis_result/cccp3_slc_map/saliency_map_of_No'+ str(k) + '.png')

elif command == 231:
    for k in range(384):
        z = np.zeros(9)
        idx = np.zeros(9)
        f = open('img_name_imageNet_test.txt')
        data1 = f.read()
        f.close()
        lines1 = data1.split('\n')
        for i in range(len(lines1)-1):
            image = caffe.io.load_image(lines1[i])
            transformed_image = transformer.preprocess('data', image)
            # copy the image data into the memory allocated for the net
            net.blobs['data'].data[...] = transformed_image
            ### perform classification
            output = net.forward()
            hoge = net.blobs['cccp3'].data[4,:,:,:]
            if np.max(hoge[k,:,:]) > np.min(z):
                idx[np.argmin(z)] = i
                z[np.argmin(z)] = np.max(hoge[k,:,:])

        no_feature_map = k
        print 'map No:', no_feature_map
    
        for j in range(9):
            image = caffe.io.load_image(lines1[int(idx[j])])
            transformed_image = transformer.preprocess('data', image)
            # copy the image data into the memory allocated for the net
            net.blobs['data'].data[...] = transformed_image
            ### perform classification
            output = net.forward()
            boundary, max_idx = get_crop_image_cccp3(net, no_feature_map)
            sali_map = make_saliency_map_cccp3(net,no_feature_map, lines1[int(idx[j])],boundary,max_idx)
            plt.subplot(3,3,j)
            #plt.imshow(sali_map,aspect='auto');plt.axis('off')
            plt.imshow(sali_map);plt.axis('off')
            
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.05)
        plt.savefig('vis_result/cccp3_slc_map/saliency_map_of_No'+ str(k) + '.png')

        for j in range(9):
            image = caffe.io.load_image(lines1[int(idx[j])])
            transformed_image = transformer.preprocess('data', image)
            # copy the image data into the memory allocated for the net
            net.blobs['data'].data[...] = transformed_image
            ### perform classification
            output = net.forward()
            boundary, max_idx = get_crop_image_cccp3(net, no_feature_map)
            xmin = max(0, boundary[0]*4);
            xmax = boundary[1]*4
            ymin = max(0,boundary[2]*4);
            ymax = boundary[3]*4
            img = np.transpose(net.blobs['data'].data[4,:,:,:],(1,2,0))
            img = (img-np.min(img))/(np.max(img)-np.min(img))
            img = img[:,:,::-1]
            plt.subplot(3,3,j)
            #plt.imshow(img[xmin:xmax,ymin:ymax,:],aspect='auto');plt.axis('off')
            plt.imshow(img[xmin:xmax,ymin:ymax,:]);plt.axis('off')
            
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.05)
        plt.savefig('vis_result/cccp3_slc_map/crop_img_of_No'+ str(k) + '.png')


elif command == 220:
    for k in range(384):
        z = np.zeros(9)
        idx = np.zeros(9)
        f = open('img_name_imageNet_test.txt')
        data1 = f.read()
        f.close()
        lines1 = data1.split('\n')
        for i in range(len(lines1)-1):
            image = caffe.io.load_image(lines1[i])
            transformed_image = transformer.preprocess('data', image)
            # copy the image data into the memory allocated for the net
            net.blobs['data'].data[...] = transformed_image
            ### perform classification
            output = net.forward()
            hoge = net.blobs['conv2'].data[4,:,:,:]
            if np.max(hoge[k,:,:]) > np.min(z):
                idx[np.argmin(z)] = i
                z[np.argmin(z)] = np.max(hoge[k,:,:])

        no_feature_map = k
        print 'map No:', no_feature_map
    
        for j in range(9):
            image = caffe.io.load_image(lines1[int(idx[j])])
            transformed_image = transformer.preprocess('data', image)
            # copy the image data into the memory allocated for the net
            net.blobs['data'].data[...] = transformed_image
            ### perform classification
            output = net.forward()
            boundary, max_idx = get_crop_image_conv2(net, no_feature_map)
            sali_map = make_saliency_map_conv2(net,no_feature_map, lines1[int(idx[j])],boundary,max_idx)
            plt.subplot(3,3,j)
            #plt.imshow(sali_map,aspect='auto');plt.axis('off')
            plt.imshow(sali_map);plt.axis('off')
            
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.05)
        plt.savefig('vis_result/conv2_slc_map/saliency_map_of_No'+ str(k) + '.png')

        for j in range(9):
            image = caffe.io.load_image(lines1[int(idx[j])])
            transformed_image = transformer.preprocess('data', image)
            # copy the image data into the memory allocated for the net
            net.blobs['data'].data[...] = transformed_image
            ### perform classification
            output = net.forward()
            boundary, max_idx = get_crop_image_conv2(net, no_feature_map)
            xmin = max(0, boundary[0]*4);
            xmax = boundary[1]*4
            ymin = max(0,boundary[2]*4);
            ymax = boundary[3]*4
            img = np.transpose(net.blobs['data'].data[4,:,:,:],(1,2,0))
            img = (img-np.min(img))/(np.max(img)-np.min(img))
            img = img[:,:,::-1]
            plt.subplot(3,3,j)
            #plt.imshow(img[xmin:xmax,ymin:ymax,:],aspect='auto');plt.axis('off')
            plt.imshow(img[xmin:xmax,ymin:ymax,:]);plt.axis('off')
            
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.05)
        plt.savefig('vis_result/conv2_slc_map/crop_img_of_No'+ str(k) + '.png')

elif command == 120:
    for k in range(384):
        z = np.zeros(9)
        idx = np.zeros(9)
        f = open('img_name_imageNet_test.txt')
        data1 = f.read()
        f.close()
        lines1 = data1.split('\n')
        for i in range(len(lines1)-1):
            image = caffe.io.load_image(lines1[i])
            transformed_image = transformer.preprocess('data', image)
            # copy the image data into the memory allocated for the net
            net.blobs['data'].data[...] = transformed_image
            ### perform classification
            output = net.forward()
            hoge = net.blobs['conv2'].data[4,:,:,:]
            if np.max(hoge[k,:,:]) > np.min(z):
                idx[np.argmin(z)] = i
                z[np.argmin(z)] = np.max(hoge[k,:,:])

        no_feature_map = k
        print 'map No:', no_feature_map
    
        for j in range(9):
            image = caffe.io.load_image(lines1[int(idx[j])])
            transformed_image = transformer.preprocess('data', image)
            # copy the image data into the memory allocated for the net
            net.blobs['data'].data[...] = transformed_image
            ### perform classification
            output = net.forward()
            boundary, max_idx = get_crop_image_conv2(net, no_feature_map)
            sali_map = make_saliency_map_conv2(net,no_feature_map, lines1[int(idx[j])],boundary,max_idx)
            plt.subplot(3,3,j)
            #plt.imshow(sali_map,aspect='auto');plt.axis('off')
            plt.imshow(sali_map);plt.axis('off')
            
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.05)
        plt.savefig('vis_result/conv2_slc_map/saliency_map_of_No'+ str(k) + '.png')
