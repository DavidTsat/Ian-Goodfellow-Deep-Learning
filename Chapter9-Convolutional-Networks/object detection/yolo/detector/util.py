import torch
import numpy as np

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    # grid_size = inp_dim // stride
    grid_size = prediction.size(2)
    bbox_attrs = 5 + num_classes 
    num_anchors = len(anchors)
    
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
    
    #Sigmoid the centre_X, centre_Y. and object confidence
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    
    #Add the center offests
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)
    
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1,2).unsqueeze(0)
    
    prediction[:,:,:2] += x_y_offset
    
    #log space transform height and the width
    
    anchors = torch.FloatTensor(anchors)
    
    if CUDA:
        anchors = anchors.cuda()
    
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    
    #Apply sigmoid to the class scores
    prediction[:,:,5:5+num_classes] = torch.sigmoid((prediction[:,:,5:5+num_classes]))
    
    prediction[:,:,:4] *= stride
    
    return prediction

def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
    '''
    transform the center x, center y, height, width coordinates to the top-left corner x, 
    top-left corner y, right-bottom corner x, right-bottom corner y
    '''
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = prediction[:,:,0] - prediction[:,:,2]/2
    box_corner[:,:,1] = prediction[:,:,1] - prediction[:,:,3]/2
    box_corner[:,:,2] = prediction[:,:,0] + prediction[:,:,2]/2
    box_corner[:,:,3] = prediction[:,:,1] + prediction[:,:,3]/2
    prediction[:,:,:4] = box_corner[:,:,:4]

    batch_size = prediction.size(0)
    write = False
    for ind in range(batch_size):
        image_pred = prediction[ind]
        #confidence threshholding
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        non_zero_ind = torch.nonzero((image_pred[:,4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue
        if image_pred_.shape[0] == 0:
            continue

        #Get the various classes detected in the image
        img_classes = unique(image_pred_[:,-1]) #-1 is the class index

        #NMS
        for cls in img_classes:
            #Extract the detections of a particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            #Sort the detetctions such that the entry with the maximum objectness confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0) #number of detections

            #NMS
            for i in range(idx):
                #Get the IOUs of all boxes that come after the one we are looking at in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
                except IndexError:
                    break
                #Zero out all the detections that have IoU > threshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask
                #Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(
                ind)  # Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))
    try:
        return output
    except:
        return 0

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def bbox_iou(box1, box2):
    '''
    :param box1:
    :param box2:
    :return: Returns the IoU of two bounding boxes
    '''
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,1], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    #Get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    #Intersection area
    inter_area = torch.clamp(inter_rect_x2-inter_rect_x1+1, min=0)* torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    #Union area
    b1_area = (b1_x2-b1_x1+1)*(b2_y2-b1_y1+1)
    b2_area = (b2_x2-b2_x1+1)*(b2_y2-b2_y1+1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou