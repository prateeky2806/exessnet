import torch
import torch.nn as nn

from args import args
from .builder import Builder
from transformers import AutoModel
from torchtext.vocab import GloVe, FastText
from torch.nn.functional import avg_pool1d
import torch.nn as nn


class MaskedConvBlock(nn.Module):

    def __init__(self, builder, input_dim=128, n_filters=256, kernel_size=3, padding=1, stride=1, shortcut=False,
                 downsampling=None):
        super(MaskedConvBlock, self).__init__()

        self.downsampling = downsampling
        self.shortcut = shortcut

        self.conv1 = builder.conv_layer(input_dim, n_filters, kernel_size, padding=padding, stride=stride, bias=False)
        builder.conv_init(self.conv1)
        self.batchnorm1 = builder.batchnorm(n_filters)
        self.relu1 = nn.ReLU()
        self.conv2 = builder.conv_layer(n_filters, n_filters, kernel_size, padding=padding, stride=stride, bias=False)
        builder.conv_init(self.conv2)
        self.batchnorm2 = builder.batchnorm(n_filters)
        self.relu2 = nn.ReLU()

    def forward(self, input):

        residual = input
        output = self.conv1(input)
        output = self.batchnorm1(output)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.batchnorm2(output)

        if self.shortcut:
            if self.downsampling is not None:
                residual = self.downsampling(input)
            output += residual

        output = self.relu2(output)
        return output


class VDCNN(nn.Module):

    def __init__(self, num_classes, embedding_dim, depth=9, shortcut=False, base_num_features=64):
        super(VDCNN, self).__init__()
        self.num_classes = num_classes
        layers = []
        builder = Builder()        
        
        con = builder.conv_layer(embedding_dim, base_num_features, kernel_size=3, padding=1, bias=False)
        builder.conv_init(con)
        layers.append(con)

        if depth == 9:
            num_conv_block = [0, 0, 0, 0]
        elif depth == 17:
            num_conv_block = [1, 1, 1, 1]
        elif depth == 29:
            num_conv_block = [4, 4, 1, 1]
        elif depth == 49:
            num_conv_block = [7, 7, 4, 2]

        layers.append(MaskedConvBlock(builder, input_dim=base_num_features, n_filters=base_num_features, kernel_size=3, padding=1,
                                shortcut=shortcut))
        
        for _ in range(num_conv_block[0]):
            layers.append(MaskedConvBlock(builder, input_dim=base_num_features, n_filters=base_num_features, kernel_size=3, padding=1,
                                    shortcut=shortcut))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        ds = nn.Sequential(builder.conv_layer(base_num_features, 2 * base_num_features, kernel_size=1, stride=1, bias=False),
                           builder.batchnorm(2 * base_num_features))
        builder.conv_init(ds[0])
        layers.append(
            MaskedConvBlock(builder, input_dim=base_num_features, n_filters=2 * base_num_features, kernel_size=3, padding=1,
                      shortcut=shortcut, downsampling=ds))
        
        for _ in range(num_conv_block[1]):
            layers.append(
                MaskedConvBlock(builder, input_dim=2 * base_num_features, n_filters=2 * base_num_features, kernel_size=3, padding=1,
                          shortcut=shortcut))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        ds = nn.Sequential(builder.conv_layer(2 * base_num_features, 4 * base_num_features, kernel_size=1, stride=1, bias=False),
                           builder.batchnorm(4 * base_num_features))
        builder.conv_init(ds[0])
        layers.append(
            MaskedConvBlock(builder, input_dim=2 * base_num_features, n_filters=4 * base_num_features, kernel_size=3, padding=1,
                      shortcut=shortcut, downsampling=ds))
        
        for _ in range(num_conv_block[2]):
            layers.append(
                MaskedConvBlock(builder, input_dim=4 * base_num_features, n_filters=4 * base_num_features, kernel_size=3, padding=1,
                          shortcut=shortcut))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        ds = nn.Sequential(builder.conv_layer(4 * base_num_features, 8 * base_num_features, kernel_size=1, stride=1, bias=False),
                           builder.batchnorm(8 * base_num_features))
        builder.conv_init(ds[0])

        layers.append(
            MaskedConvBlock(builder, input_dim=4 * base_num_features, n_filters=8 * base_num_features, kernel_size=3, padding=1,
                      shortcut=shortcut, downsampling=ds))
        for _ in range(num_conv_block[3]):
            layers.append(
                MaskedConvBlock(builder, input_dim=8 * base_num_features, n_filters=8 * base_num_features, kernel_size=3, padding=1,
                          shortcut=shortcut))

        layers.append(nn.AdaptiveMaxPool1d(8))
        self.layers = nn.Sequential(*layers)
        self.linear = builder.conv1x1(8 * base_num_features, num_classes, last_layer=True)

    def forward(self, input, reps=False):
        output = input.transpose(1, 2)
        output = self.layers(output)
        output = avg_pool1d(output, 8)
        if reps:
            return output
        output = self.linear(output)
        return output


class CNNStatic(nn.Module):
    def __init__(self, num_classes, embeddim, maxlen, numfilters, filtersizes):
        super(CNNStatic,self).__init__()

        self.numfilters = numfilters
        self.filtersizes = filtersizes
        self.embeddim = embeddim

        builder = Builder()
        
        self.conv1 = builder.conv_layer(self.embeddim, self.numfilters, kernel_size=self.filtersizes[0], bias=False)
        builder.conv_init(self.conv1)
        self.pool1 = nn.MaxPool1d(kernel_size=(maxlen-self.filtersizes[0]+1))
        
        self.conv2 = builder.conv_layer(self.embeddim, self.numfilters, kernel_size=self.filtersizes[1], bias=False)
        builder.conv_init(self.conv2)
        self.pool2 = nn.MaxPool1d(kernel_size=(maxlen-self.filtersizes[1]+1))
        
        self.conv3 = builder.conv_layer(self.embeddim, self.numfilters, kernel_size=self.filtersizes[2], bias=False)
        builder.conv_init(self.conv3)
        self.pool3 = nn.MaxPool1d(kernel_size=(maxlen-self.filtersizes[2]+1))
        
        self.act = nn.ReLU()
        self.linear = builder.conv1x1(self.numfilters * 3, num_classes, last_layer=True)
        
        # self.drop = nn.Dropout(0)

    def forward(self, x, reps=False):
        x = x.transpose(1,2)
        out1 = self.pool1(self.act(self.conv1(x)))
        out2 = self.pool2(self.act(self.conv2(x)))
        out3 = self.pool3(self.act(self.conv3(x)))
        out = torch.cat([out1,out2,out3],dim=1)
        if reps:
            return out
        out = self.linear(out)
        return out


class TextCLModel(nn.Module):
    def __init__(self, num_classes):
        super(TextCLModel, self).__init__()
        self.numfilters = args.num_filters
        self.filtersizes = [3, 4, 5]
        if args.emb_model in ['glove', 'fasttext']:
            self.hidden_size = 300
            if args.emb_model == 'glove':
                glove_vectors = GloVe(name="42B", dim=self.hidden_size, max_vectors=40000).vectors
            elif args.emb_model == 'fasttext':
                glove_vectors = FastText(max_vectors=40000).vectors
            glove_vectors = torch.cat((torch.zeros(1, self.hidden_size), torch.randn((1, self.hidden_size)), glove_vectors))
            self.pretrained_embeddings = torch.nn.Embedding.from_pretrained(glove_vectors, freeze=True, padding_idx=1, sparse=False) 
        else:
            self.pretrained_embeddings = AutoModel.from_pretrained(args.emb_model)
            self.hidden_size = self.pretrained_embeddings.config.hidden_size

        if args.cnn_model == "vdcnn":
            base_num_feat = 64
            self.cnn_model = VDCNN(num_classes, self.hidden_size, depth=args.vdcnn_depth, shortcut=False, base_num_features=base_num_feat)
            # lin_in = 8 * 8 * base_num_feat
        elif args.cnn_model == "cnnstatic":
            self.cnn_model = CNNStatic(num_classes, self.hidden_size, args.max_length,
                                        self.numfilters, self.filtersizes)
            # lin_in = self.numfilters*3

        # self.classifier = nn.Sequential(nn.Linear(lin_in, args.lin_hidden), nn.ReLU(), 
        #                 nn.Linear(args.lin_hidden, args.lin_hidden), nn.ReLU(),
        #                 nn.Linear(args.lin_hidden, num_classes))

        print('hehe')

    def forward(self, x, mask, reps=False, token_type_ids=None):
        if args.emb_model in ['glove', 'fasttext']:
            word_embeddings = self.pretrained_embeddings(x)
            pass
        else:
            output = self.pretrained_embeddings(input_ids=x,
                            attention_mask=mask,
                            token_type_ids=token_type_ids,
                            return_dict=True)
            word_embeddings = output.last_hidden_state

        cnn_embeddings = self.cnn_model(word_embeddings, reps)
        preds = cnn_embeddings.squeeze(-1)
        # preds = self.classifier(cnn_embeddings)
        return preds
