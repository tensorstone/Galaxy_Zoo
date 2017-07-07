import csv
f=open('/Users/sunhop/Desktop/Galaxy data/Numb.csv')
reader = csv.reader(f)
for row in reader:
    p = '/Users/sunhop/Desktop/Galaxy data/images_training_rev1/'+str(row[0])+'.jpg'     
    data=Image.open(p)
    plt.imshow(data)
    xsize,ysize=data.size
    #print(xsize,ysize)
    box=(142,142,282,282)
    roi=data.crop(box)
    #plt.imshow(roi)
    roi.save('/Users/sunhop/Desktop/tdata/VAE_data/'+str(row[0])+'.jpg')
    
# use code above to cut the original figures into 140*140*3(RGB)
