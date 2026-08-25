import torch
import torch.nn as nn
from torch.amp import autocast
from typing import Dict, Union, Tuple, Optional
from vae.base import BaseVAE, BaseDecoder, BaseEncoder
from collections import OrderedDict


class CVAEMemRandEncoder(BaseEncoder):
    def __init__(self, config: dict):
        '''
            Encoder for the main observation data.
            
            Given X=(B,T,H,W) the surface sequence and A=(B,T,n) the extra information, the encoder does the following:
            - Embed surface into (B,T,n_surface)
            - Embed extra features into (B,T,n_info) (Not necessary for now)
            - Concate both info, (B,T,n_surface+n_info)
            - Encode time features using memory (RNN/GRU/LSTM), (B,T,n_mem)
            - Map to latent space, (B, T, latent_dim), latents will be generated for each timestep, 
            as we don't want future information to be observed for current/previous timesteps
        '''
        super(CVAEMemRandEncoder, self).__init__(config)

        latent_dim = config["latent_dim"]

        surface_embedding_final_dim = self.__get_surface_embedding(config)
        if config["ex_feats_dim"] > 0:
            ex_feats_embedding_final_dim = self.__get_ex_feats_embedding(config)
        else:
            ex_feats_embedding_final_dim = 0
        
        self.__get_interaction_layers(config, surface_embedding_final_dim + ex_feats_embedding_final_dim)
        mem_final_dim = self.__get_mem(config, surface_embedding_final_dim + ex_feats_embedding_final_dim)

        self.z_mean_layer = nn.Linear(mem_final_dim, latent_dim)
        self.z_log_var_layer = nn.Linear(mem_final_dim, latent_dim)

    def __get_surface_embedding(self, config):
        '''Embedding for vol surface'''
        feat_dim = config["feat_dim"]

        surface_embedding_layers = config["surface_hidden"]

        surface_embedding = OrderedDict()
        if config["use_dense_surface"]:
            in_feats = feat_dim[0] * feat_dim[1]
            surface_embedding["flatten"] = nn.Flatten()
            for i, out_feats in enumerate(surface_embedding_layers):
                surface_embedding[f"enc_dense_{i}"] = nn.Linear(in_feats, out_feats)
                surface_embedding[f"enc_activation_{i}"] = nn.ReLU()
                in_feats = out_feats
            
            self.surface_embedding = nn.Sequential(surface_embedding)
            final_dim = in_feats
        else:
            # convolutional layer
            padding = config["padding"]
            in_feats = 1 # encoding per vol surface
            for i, out_feats in enumerate(surface_embedding_layers):
                
                surface_embedding[f"enc_conv_{i}"] = nn.Conv2d(
                    in_feats, out_feats,
                    kernel_size=3, stride=1, padding=padding,
                )
                surface_embedding[f"enc_activation_{i}"] = nn.ReLU()
                in_feats = out_feats
            surface_embedding["flatten"] = nn.Flatten()
            self.surface_embedding = nn.Sequential(surface_embedding)

            final_dim = in_feats * feat_dim[0] * feat_dim[1]
        return final_dim

    def __get_ex_feats_embedding(self, config):
        '''Embedding for extra features'''
        ex_feats_dim = config["ex_feats_dim"]
        ex_feats_embedding_layers = config["ex_feats_hidden"]
        if ex_feats_embedding_layers is None:
            # we simply assume that there is no need for embedding the extra features
            self.ex_feats_embedding = nn.Identity()
            return ex_feats_dim
        
        # The following code is not used and not tested
        ex_feats_embedding = OrderedDict()
        in_feats = ex_feats_dim
        for i, out_feats in enumerate(ex_feats_embedding_layers):
            ex_feats_embedding[f"ex_enc_linear_{i}"] = nn.Linear(in_feats, out_feats)
            ex_feats_embedding[f"ex_enc_activation_{i}"] = nn.ReLU()
            in_feats = out_feats
        self.ex_feats_embedding = nn.Sequential(ex_feats_embedding)
        return in_feats # final dimension

    def __get_interaction_layers(self, config, input_size):
        if config["interaction_layers"] is not None and config["interaction_layers"] > 0:
            num_layers = config["interaction_layers"]
            interaction = OrderedDict()
            for i in range(num_layers):
                interaction[f"enc_interact_linear_{i}"] = nn.Linear(input_size, input_size)
                interaction[f"enc_interaction_activation_{i}"] = nn.ReLU()
            
            interaction["enc_interact_linear_final"] = nn.Linear(input_size, input_size)
            self.interaction = nn.Sequential(interaction)
        else:
            self.interaction = nn.Identity()

    def __get_mem(self, config, input_size):
        # Memory using LSTM/RNN/GRU
        mem_type = config["mem_type"]
        mem_args = {
            "input_size": input_size,
            "hidden_size": config["mem_hidden"],
            "num_layers": config["mem_layers"],
            "batch_first": True,
            "dropout": config["mem_dropout"],
        }
        if mem_type == "lstm":
            self.mem = nn.LSTM(**mem_args)
        elif mem_type == "gru":
            self.mem = nn.GRU(**mem_args)
        else:
            self.mem = nn.RNN(**mem_args)
        return config["mem_hidden"]

    def forward(self, x):
        '''
            Input:
                x should be a dictionary with 2 keys:
                - surface: volatility surface of shape (B,T,H,W), 
                    surface[:,:context_len,:,:] are the context surfaces (previous days),
                    surface[:,context_len:,:,:] is the surface to predict
                - ex_feats: extra features of shape (B,T,n), this doesn't have to exist
        '''
        latent_dim = self.config["latent_dim"]
        surface: torch.Tensor = x["surface"] # (B, T, H, W)
        T = surface.shape[1]

        # Embed surface of each day individually
        surface = surface.reshape((surface.shape[0] * surface.shape[1], 1, surface.shape[2], surface.shape[3])) # (BxT, 1, H, W), this works for both dense and conv versions
        surface_embedding = self.surface_embedding(surface) # (BxT, surface_embedding_final_dim)
        surface_embedding = surface_embedding.reshape((-1, T, surface_embedding.shape[1])) # (B, T, surface_embedding_final_dim)
        
        if "ex_feats" in x:
            ex_feats: torch.Tensor = x["ex_feats"] # (B, T, n)
            # Embed features of each day individually
            ex_feats = ex_feats.reshape((ex_feats.shape[0] * ex_feats.shape[1], ex_feats.shape[2])) # (BxT, n)
            ex_feats_embedding = self.ex_feats_embedding(ex_feats) # (BxT, ex_feats_embedding_final_dim)
            ex_feats_embedding = ex_feats_embedding.reshape((-1, T, ex_feats_embedding.shape[1])) # (B, T, ex_feats_embedding_final_dim)
        
            # concat the embeddings and get the time features
            embeddings = torch.cat([surface_embedding, ex_feats_embedding], dim=-1)  # (B, T, surface_embedding_final_dim + ex_feats_embedding_final_dim)
        else:
            embeddings = surface_embedding

        embeddings = self.interaction(embeddings) # add some nonlinear interactions, nn layers act on the final dimension only
        embeddings, _ = self.mem(embeddings) # (B, T, n_lstm)

        embeddings = embeddings.reshape((embeddings.shape[0] * embeddings.shape[1], embeddings.shape[2])) # (BxT, n_lstm)
        z_mean = self.z_mean_layer(embeddings).reshape((-1, T, latent_dim)) # (B, T, latent_dim)
        z_log_var = self.z_log_var_layer(embeddings).reshape((-1, T, latent_dim)) # (B, T, latent_dim)
        eps = torch.randn_like(z_log_var)
        z = z_mean + torch.exp(0.5 * z_log_var) * eps
        return (z_mean, z_log_var, z)

class CVAECtxMemRandEncoder(BaseEncoder):
    def __init__(self, config: dict):
        '''
            Encoder for the context.
            
            Given X=(B,C,H,W) the surface sequence and A=(B,C,n) the extra information, the context encoder does the following:
            - Embed surface into (B,C,n_surface)
            - Embed extra features into (B,C,n_info) (Not necessary for now)
            - Concate both info, (B,C,n_surface+n_info)
            - Encode time features using LSTM, (B,C,n_lstm)
            - (Potentially) compress to (B,C,latent_dim)
        '''
        super(CVAECtxMemRandEncoder, self).__init__(config)

        # There is no need for distribution sampling
        surface_embedding_dim = self.__get_surface_embedding(config)
        if config["ex_feats_dim"] > 0:
            ex_feats_embedding_dim = self.__get_ex_feats_embedding(config)
        else:
            ex_feats_embedding_dim = 0
        
        self.__get_interaction_layers(config, surface_embedding_dim + ex_feats_embedding_dim)
        mem_hidden = self.__get_mem(config, surface_embedding_dim + ex_feats_embedding_dim)
        if config["compress_context"]:
            self.final_compression = nn.Linear(mem_hidden, config["latent_dim"])
        else:
            self.final_compression = nn.Identity()
    
    def __get_surface_embedding(self, config):
        '''Embedding for vol surface'''
        feat_dim = config["feat_dim"]

        ctx_surface_embedding_layers = config["ctx_surface_hidden"]
        
        ctx_surface_embedding = OrderedDict()
        if config["use_dense_surface"]:
            in_feats = feat_dim[0] * feat_dim[1]
            ctx_surface_embedding["flatten"] = nn.Flatten()
            for i, out_feats in enumerate(ctx_surface_embedding_layers):
                ctx_surface_embedding[f"ctx_enc_dense_{i}"] = nn.Linear(in_feats, out_feats)
                ctx_surface_embedding[f"ctx_enc_activation_{i}"] = nn.ReLU()
                in_feats = out_feats
            
            self.ctx_surface_embedding = nn.Sequential(ctx_surface_embedding)
            final_dim = in_feats
        else:
            padding = config["padding"]
            in_feats = 1 # encoding per vol surface
            for i, out_feats in enumerate(ctx_surface_embedding_layers):
                
                ctx_surface_embedding[f"ctx_enc_conv_{i}"] = nn.Conv2d(
                    in_feats, out_feats,
                    kernel_size=3, stride=1, padding=padding,
                )
                ctx_surface_embedding[f"ctx_enc_activation_{i}"] = nn.ReLU()
                in_feats = out_feats
            ctx_surface_embedding["flatten"] = nn.Flatten()
            self.ctx_surface_embedding = nn.Sequential(ctx_surface_embedding)

            final_dim = in_feats * feat_dim[0] * feat_dim[1]
        return final_dim

    def __get_ex_feats_embedding(self, config):
        '''Embedding for extra features'''
        ex_feats_dim = config["ex_feats_dim"]
        ctx_ex_feats_embedding_layers = config["ctx_ex_feats_hidden"]
        if ctx_ex_feats_embedding_layers is None:
            # we simply assume that there is no need for embedding the extra features
            self.ctx_ex_feats_embedding = nn.Identity()
            return ex_feats_dim
        
        # The following code is not used and not tested
        ctx_ex_feats_embedding = OrderedDict()
        in_feats = ex_feats_dim
        for i, out_feats in enumerate(ctx_ex_feats_embedding_layers):
            ctx_ex_feats_embedding[f"ctx_ex_enc_linear_{i}"] = nn.Linear(in_feats, out_feats)
            ctx_ex_feats_embedding[f"ctx_ex_enc_activation_{i}"] = nn.ReLU()
            in_feats = out_feats
        self.ctx_ex_feats_embedding = nn.Sequential(ctx_ex_feats_embedding)
        return in_feats # final dimension
    
    def __get_interaction_layers(self, config, input_size):
        if config["interaction_layers"] is not None and config["interaction_layers"] > 0:
            num_layers = config["interaction_layers"]
            interaction = OrderedDict()
            for i in range(num_layers):
                interaction[f"ctx_interact_linear_{i}"] = nn.Linear(input_size, input_size)
                interaction[f"ctx_interaction_activation_{i}"] = nn.ReLU()
            
            interaction["ctx_interact_linear_final"] = nn.Linear(input_size, input_size)
            self.interaction = nn.Sequential(interaction)
        else:
            self.interaction = nn.Identity()

    def __get_mem(self, config, input_size):
        # Memory using LSTM/RNN/GRU
        mem_type = config["mem_type"]
        mem_args = {
            "input_size": input_size,
            "hidden_size": config["mem_hidden"],
            "num_layers": config["mem_layers"],
            "batch_first": True,
            "dropout": config["mem_dropout"],
        }
        if mem_type == "lstm":
            self.mem = nn.LSTM(**mem_args)
        elif mem_type == "gru":
            self.mem = nn.GRU(**mem_args)
        else:
            self.mem = nn.RNN(**mem_args)
        return config["mem_hidden"]

    def forward(self, x):
        '''
            Input:
                x should be a dictionary with 2 keys:
                - surface: volatility surface of shape (B,C,H,W), 
                    surface[:,:context_len,:,:] are the context surfaces (previous days),
                    surface[:,context_len:,:,:] is the surface to predict
                - ex_feats: extra features of shape (B,C,n)
        '''
        surface: torch.Tensor = x["surface"] # (B, C, H, W)
        C = surface.shape[1]

        # Embed surface of each day individually
        surface = surface.reshape((surface.shape[0] * surface.shape[1], 1, surface.shape[2], surface.shape[3])) # (BxC, 1, H, W), this works for both dense and conv versions
        surface_embedding = self.ctx_surface_embedding(surface)
        surface_embedding = surface_embedding.reshape((-1, C, surface_embedding.shape[1]))

        if "ex_feats" in x:
            ex_feats: torch.Tensor = x["ex_feats"] # (B, C, n)
            # Embed features of each day individually
            ex_feats = ex_feats.reshape((ex_feats.shape[0] * ex_feats.shape[1], ex_feats.shape[2])) # (BxT, n)
            ex_feats_embedding = self.ctx_ex_feats_embedding(ex_feats) # (BxT, ex_feats_embedding_final_dim)
            ex_feats_embedding = ex_feats_embedding.reshape((-1, C, ex_feats_embedding.shape[1])) # (B, T, ex_feats_embedding_final_dim)

            ctx_embeddings = torch.cat([surface_embedding, ex_feats_embedding], dim=-1)
        else:
            ctx_embeddings = surface_embedding
        ctx_embeddings = self.interaction(ctx_embeddings) # linear acts on the final layers only
        ctx_embeddings, _ = self.mem(ctx_embeddings)
        ctx_embeddings = self.final_compression(ctx_embeddings)
        return ctx_embeddings

class CVAEMemRandDecoder(BaseDecoder):
    def __init__(self, config: dict):
        '''
            Inputs to this module: 1. the latents generated by main encoder on seq_len input. (B, T, latent_dim) 2. the embedding generated by context encoder on context_len input (B, C, mem_hidden) padded with zeros to (B, T, mem_hidden)
            
            The decoder does the following:
            - Memory map back to (B, T, n_surface+n_info)
            - Deconv on (B, T, n_surface) to reconstruct the surface
            - Dense mapping on (B, T, n_info) to reconstruct the extra features
        '''

        super(CVAEMemRandDecoder, self).__init__(config)

        surface_embedding_layers = config["surface_hidden"]
        ex_feats_embedding_layers = config["ex_feats_hidden"]
        feat_dim = config["feat_dim"]
        latent_dim = config["latent_dim"]
        if config["compress_context"]:
            ctx_embedding_dim = config["latent_dim"]
        else:
            ctx_embedding_dim = config["mem_hidden"]

        # record the final hidden size for forward function
        self.surface_final_hidden_size = surface_embedding_layers[-1]
        if ex_feats_embedding_layers is not None:
            self.n_info = self.ex_feats_final_hidden_size = ex_feats_embedding_layers[-1]  
        else:
            self.n_info = self.ex_feats_final_hidden_size = config["ex_feats_dim"]
        
        if config["use_dense_surface"]:
            self.n_surface = surface_embedding_layers[-1]
        else:
            self.n_surface = surface_embedding_layers[-1] * feat_dim[0] * feat_dim[1]

        self.__get_mem(config, latent_dim + ctx_embedding_dim, self.n_surface + self.n_info)

        self.__get_interaction_layers(config, self.n_surface + self.n_info)
        self.surface_decoder_input = nn.Linear(self.n_surface + self.n_info, self.n_surface)
        self.__get_surface_decoder(config)

        if self.n_info > 0:
            self.ex_feats_decoder_input = nn.Linear(self.n_surface + self.n_info, self.n_info)
            self.__get_ex_feats_decoder(config)

    def __get_surface_decoder(self, config):
        surface_embedding_layers = config["surface_hidden"]

        surface_decoder = OrderedDict()
        if config["use_dense_surface"]:
            feat_dim = config["feat_dim"]
            in_feats = surface_embedding_layers[-1]
            for i, out_feats in enumerate(reversed(surface_embedding_layers[:-1])):
                surface_decoder[f"dec_dense_{i}"] = nn.Linear(in_feats, out_feats)
                surface_decoder[f"dec_activation_{i}"] = nn.ReLU()
                in_feats = out_feats
            # transform to the original size
            final_size = feat_dim[0] * feat_dim[1]
            surface_decoder["dec_final"] = nn.Linear(in_feats, final_size)
            surface_decoder["dec_final_activation"] = nn.ReLU()
            # Output layer (single surface prediction)
            surface_decoder["dec_output"] = nn.Linear(final_size, final_size)
        else:
            padding = config["padding"]
            deconv_output_padding = config["deconv_output_padding"]
            in_feats = surface_embedding_layers[-1]
            for i, out_feats in enumerate(reversed(surface_embedding_layers[:-1])):
                surface_decoder[f"dec_deconv_{i}"] = nn.ConvTranspose2d(
                    in_feats, out_feats,
                    kernel_size=3, stride=1, padding=padding, output_padding=deconv_output_padding,
                    )
                surface_decoder[f"dec_activation_{i}"] = nn.ReLU()
                in_feats = out_feats
            
            # transform to the original size
            surface_decoder["dec_final"] = nn.ConvTranspose2d(
                in_feats, in_feats,
                kernel_size=3, stride=1, padding=padding, output_padding=deconv_output_padding,
            )
            surface_decoder["dec_final_activation"] = nn.ReLU()
            # Output layer (single channel surface prediction)
            surface_decoder["dec_output"] = nn.Conv2d(
                in_feats, 1,
                kernel_size=3, padding="same"
            )
        self.surface_decoder = nn.Sequential(surface_decoder)

    def __get_ex_feats_decoder(self, config):
        ex_feats_dim = config["ex_feats_dim"]
        ex_feats_embedding_layers = config["ex_feats_hidden"]
        if ex_feats_embedding_layers is None:
            # we simply assume that there is no need for embedding the extra features, but we need a linear mapping now
            self.ex_feats_decoder = nn.Linear(ex_feats_dim, ex_feats_dim)
            return
        
        # The following code is not used and not tested
        ex_feats_decoder = OrderedDict()
        in_feats = ex_feats_embedding_layers[-1]
        for i, out_feats in enumerate(reversed(ex_feats_embedding_layers[:-1])):
            ex_feats_decoder[f"ctx_ex_dec_linear_{i}"] = nn.Linear(in_feats, out_feats)
            ex_feats_decoder[f"ctx_ex_dec_activation_{i}"] = nn.ReLU()
            in_feats = out_feats
        ex_feats_decoder["ctx_dec_output"] = nn.Linear(in_feats, ex_feats_dim)
        self.ex_feats_decoder = nn.Sequential(ex_feats_decoder)
    
    def __get_interaction_layers(self, config, input_size):
        if config["interaction_layers"] is not None and config["interaction_layers"] > 0:
            num_layers = config["interaction_layers"]
            interaction = OrderedDict()
            for i in range(num_layers):
                interaction[f"dec_interact_linear_{i}"] = nn.Linear(input_size, input_size)
                interaction[f"dec_interaction_activation_{i}"] = nn.ReLU()
            
            interaction["dec_interact_linear_final"] = nn.Linear(input_size, input_size)
            self.interaction = nn.Sequential(interaction)
        else:
            self.interaction = nn.Identity()

    def __get_mem(self, config, input_size, hidden_size):
        # Memory using LSTM/RNN/GRU
        mem_type = config["mem_type"]
        mem_args = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": config["mem_layers"],
            "batch_first": True,
            "dropout": config["mem_dropout"],
        }
        if mem_type == "lstm":
            self.mem = nn.LSTM(**mem_args)
        elif mem_type == "gru":
            self.mem = nn.GRU(**mem_args)
        else:
            self.mem = nn.RNN(**mem_args)
    
    def forward(self, x):
        '''
            Input:
                x should be (B, T, latent_dim + mem_hidden), where x[:,T-C:,latent_dim:] should be 0
        '''
        feat_dim = self.config["feat_dim"]
        ex_feats_dim = self.config["ex_feats_dim"]
        # should be on device already
        x, _ = self.mem(x) # (B, T, n_surface+n_info)
        x = self.interaction(x)
        B, T = x.shape[0], x.shape[1]
        surface_x = self.surface_decoder_input(x) # (B, T, n_surface)
        if self.config["use_dense_surface"]:
            surface_x = surface_x.reshape(-1, self.surface_final_hidden_size) # (BxT, surface_final_hidden_size)
            decoded_surface = self.surface_decoder(surface_x)
            # Reshape to (B, T, H, W)
            decoded_surface = decoded_surface.reshape((B, T, feat_dim[0], feat_dim[1]))
        else:
            surface_x = surface_x.reshape(-1, self.surface_final_hidden_size, feat_dim[0], feat_dim[1]) # (BxT, surface_final_hidden_size, H, W)
            decoded_surface = self.surface_decoder(surface_x)  # (BxT, 1, H, W)
            # Reshape to (B, T, H, W)
            decoded_surface = decoded_surface.reshape((B, T, feat_dim[0], feat_dim[1]))

        if ex_feats_dim > 0:
            info_x = self.ex_feats_decoder_input(x) # (B, T, n_info)
            info_x = info_x.reshape(B*T, self.n_info) # (BxT, n_info)
            decoded_ex_feat = self.ex_feats_decoder(info_x) # (BxT, n_info)
            decoded_ex_feat = decoded_ex_feat.reshape((B, T, ex_feats_dim))

            return decoded_surface, decoded_ex_feat
        else:
            return decoded_surface

class CVAEMemRand(BaseVAE):
    def __init__(self, config: dict):
        '''
            Similar to CVAEMem in cvae_with_mem, but we don't assume anything about seq_len (T) or ctx_len (C) here.
            The input size should be (B,T,H,W), sequence length will be used as the color channel, for now, we assume T=seq_len = ctx_len(C)+1
            
            Input:
                config: must contain feat_dim, latent_dim, device, kl_weight, re_feat_weight, surface_hidden, ex_feats_dim, ex_feats_hidden, mem_type, mem_hidden, mem_layers, mem_dropout, ctx_surface_hidden, ctx_ex_feats_hidden
                feat_dim: the feature dimension of each time step, integer if 1D, tuple of integers if 2D
                latent_dim: dimension for latent space
                kl_weight: weight \beta used for loss = RE + \beta * KL, (default: 1.0)
                re_feat_weight: weight \alpha used for RE = RE(surface) + \alpha * RE(ex_feats), (default: 1.0)
                ex_feats_loss_type: loss type for RE(ex_feats), (default: l1)
                surface_hidden: hidden layer sizes for vol surface encoding
                ex_feats_dim: the number of extra features
                ex_feats_hidden: the hidden layer sizes for the extra feature (default: None, if None, identity mapping)
                mem_type: lstm/gru/rnn (default: lstm)
                mem_hidden: hidden size for memory, int
                mem_layers: number of layers for memory, int
                mem_dropout: dropout rate for memory (default: 0)
                ctx_surface_hidden: the hidden layer sizes for the context encoder (for the vol surface)
                ctx_ex_feats_hidden: the hidden layer sizes for the context encoder (for the extra features, default: None)
                interaction_layers: number of nonlinear layers for surface and extra features to interact (default: 2)
                use_dense_surface: whether or not flatten the surface into 1D and use Dense Layers for encoding/decoding (default: False)
                compress_context: whether or not compress the context encoding to the same size as the latent dimension (default: True)
                ex_loss_on_ret_only: whether or not the return should only be optimized on surface & return. Any extra features will not get loss optimized. We assume that ret is always the first ex_feature, index=0. (default: False)
        '''
        super(CVAEMemRand, self).__init__(config)
        self.check_input(config)
        if not config["use_dense_surface"]:
            # we want to keep the dimensions the same, out_dim = (in_dim - kernel_size + 2*padding) / stride + 1
            # so padding = ((out_dim -1) * stride + kernel_size - in_dim) // 2 where in_dim and out_dim are 5
            stride = 1
            feat_dim = config["feat_dim"]
            padding = ((feat_dim[-1] - 1) * stride + 3 - feat_dim[-1]) // 2
            if ((feat_dim[-1] - 1) * stride + 3 - feat_dim[-1]) % 2 == 1:
                padding += 1
                deconv_output_padding = 1
            else:
                deconv_output_padding = 0
            
            config["padding"] = padding
            config["deconv_output_padding"] = deconv_output_padding

        self.encoder = CVAEMemRandEncoder(config)
        self.ctx_encoder = CVAECtxMemRandEncoder(config)
        self.decoder = CVAEMemRandDecoder(config)

        if config["ex_feats_loss_type"] == "l2":
            self.ex_feats_loss_fn = nn.MSELoss()
        else:
            self.ex_feats_loss_fn = nn.L1Loss()

        # Move all modules to device
        self.to(self.device)

        # Multi-horizon training configuration
        self.horizon = config.get("horizon", 1)
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")

    def load_fitted_prior(self, prior_params_path: str):
        """
        Load pre-fitted aggregated posterior for realistic sampling.

        Instead of sampling future latents from N(0,1), we sample from a GMM
        fitted to the empirical aggregated posterior q(z) = E_x[q(z|x)].
        This addresses the VAE prior mismatch that causes systematic bias.

        Args:
            prior_params_path: Path to fitted GMM parameters (.pt file)
        """
        params = torch.load(prior_params_path, map_location=self.device)
        self.fitted_prior = {
            'means': params['means'].to(self.device).to(torch.float64),
            'covariances': params['covariances'].to(self.device).to(torch.float64),
            'weights': params['weights'].to(self.device).to(torch.float64),
            'n_components': params['n_components']
        }
        # Pre-compute Cholesky decomposition for efficient sampling
        self.fitted_prior['cholesky'] = torch.linalg.cholesky(self.fitted_prior['covariances'])
        print(f"Loaded fitted prior with {params['n_components']} components from {prior_params_path}")

    def _sample_from_fitted_prior(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """
        Sample latents from fitted GMM prior instead of N(0,1).

        This method samples from the empirical aggregated posterior, matching
        the distribution the decoder was trained on and eliminating systematic bias.

        Args:
            batch_size: Batch size (B)
            seq_len: Sequence length (T = context + horizon)

        Returns:
            Sampled latents (B, T, latent_dim)
        """
        if not hasattr(self, 'fitted_prior') or self.fitted_prior is None:
            raise ValueError("Fitted prior not loaded. Call load_fitted_prior() first.")

        total_samples = batch_size * seq_len
        latent_dim = self.config["latent_dim"]

        # Sample component indices based on weights
        weights = self.fitted_prior['weights']
        component_idx = torch.multinomial(
            weights.expand(total_samples, -1),
            num_samples=1
        ).squeeze(-1)  # (B*T,)

        # Get means and Cholesky factors for selected components
        means = self.fitted_prior['means'][component_idx]  # (B*T, latent_dim)
        L = self.fitted_prior['cholesky'][component_idx]   # (B*T, latent_dim, latent_dim)

        # Sample using reparameterization: z = mean + L @ eps
        eps = torch.randn(total_samples, latent_dim, device=self.device, dtype=torch.float64)
        z = means + torch.einsum('bij,bj->bi', L, eps)

        return z.reshape(batch_size, seq_len, latent_dim)

    def get_surface_given_conditions(self, c: dict[str, torch.Tensor], z: torch.Tensor=None, mu=0, std=1, horizon=None, prior_mode="standard"):
        '''
        Generate surface given context only (realistic deployment).

        Args:
            c: context dictionary with "surface" (B,C,H,W) and "ex_feats" (B,C,3)
            z: pre-generated latent samples (B,T,latent_dim). If None, uses hybrid sampling:
               - z[:, :C, :] = posterior mean from context (deterministic)
               - z[:, C:C+horizon, :] = sampled from prior (stochastic)
            mu: Mean for standard prior (only used if prior_mode="standard")
            std: Std for standard prior (only used if prior_mode="standard")
            horizon: Number of days to forecast. If None, uses self.horizon
            prior_mode: Prior sampling mode:
                - "standard": Sample from N(mu, std) - default N(0,1)
                - "fitted": Sample from fitted GMM (requires load_fitted_prior())

        Returns:
            If ex_feats present: (surf_pred, ex_pred) where:
                - surf_pred: (B, horizon, 3, 5, 5)
                - ex_pred: (B, horizon, 3)
            Otherwise: surf_pred only
        '''

        ctx_surface = c["surface"].to(self.device)
        if len(ctx_surface.shape) == 3:
            ctx_surface = ctx_surface.unsqueeze(0)
        C = ctx_surface.shape[1]
        B = ctx_surface.shape[0]

        # Use provided horizon or model's default
        if horizon is None:
            horizon = self.horizon

        T = C + horizon
        ctx = {"surface": ctx_surface}

        if "ex_feats" in c:
            ctx_ex_feats = c["ex_feats"].to(self.device)
            if len(ctx_ex_feats.shape) == 2:
                ctx_ex_feats = ctx_ex_feats.unsqueeze(0)
            assert ctx_ex_feats.shape[1] == C, "context length mismatch"
            ctx["ex_feats"] = ctx_ex_feats

        # Sample latent z based on prior_mode
        if z is not None:
            if len(z.shape) == 2:
                z = z.unsqueeze(0)
        else:
            if prior_mode == "fitted":
                z = self._sample_from_fitted_prior(B, T)
            else:
                z = mu + torch.randn((B, T, self.config["latent_dim"]),
                                     device=self.device, dtype=torch.float64) * std

        # Encode context to get posterior mean for context positions
        ctx_latent_mean, ctx_latent_log_var, ctx_latent = self.encoder(ctx)

        # Replace context positions with deterministic posterior mean
        z[:, :C, ...] = ctx_latent_mean

        ctx_embedding = self.ctx_encoder(ctx) # embedded c
        ctx_embedding_dim = ctx_embedding.shape[2]
        ctx_embedding_padded = torch.zeros((B, T, ctx_embedding_dim)).to(self.device)
        ctx_embedding_padded[:, :C, :] = ctx_embedding
        z = z.to(self.device)
        decoder_input = torch.cat([ctx_embedding_padded, z], dim=-1)
        if "ex_feats" in c:
            decoded_surface, decoded_ex_feat = self.decoder(decoder_input) # P(x|c,z,t)
            # decoded_surface: (B, T, H, W)
            return decoded_surface[:, C:, :, :], decoded_ex_feat[:, C:, :]
        else:
            decoded_surface = self.decoder(decoder_input) # P(x|c,z,t)
            # decoded_surface: (B, T, H, W)
            return decoded_surface[:, C:, :, :]

    def generate_autoregressive_sequence(
        self,
        initial_context: Dict[str, torch.Tensor],
        horizon: int = 30,
        z: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate multi-step sequence by iteratively feeding predictions back as context.

        This method supports all 3 model variants:
        - no_ex: Only surface predictions
        - ex_no_loss: Surface + ex_feats (passive conditioning)
        - ex_loss: Surface + ex_feats (joint optimization)

        Args:
            initial_context: dict with keys:
                - "surface": (B, C, 5, 5) or (C, 5, 5) - context surfaces
                - "ex_feats": optional, (B, C, 3) or (C, 3) - extra features
            horizon: int, number of days to generate (default 30)
            z: optional pre-sampled latents for reproducibility

        Returns:
            If ex_feats in context:
                tuple of (surfaces, ex_feats)
                - surfaces: (B, horizon, 5, 5) - predicted surfaces
                - ex_feats: (B, horizon, 3) - 3 features [ret, skew, slope]
            Else:
                surfaces: (B, horizon, 5, 5)
        """
        # Validate input
        if "surface" not in initial_context:
            raise ValueError("initial_context must contain 'surface' key")

        surface_shape = initial_context["surface"].shape
        if len(surface_shape) not in [3, 4]:
            raise ValueError(
                f"surface must be (C, H, W) or (B, C, H, W), got shape {surface_shape}"
            )

        # Detect if we have ex_feats
        has_ex_feats = "ex_feats" in initial_context

        if has_ex_feats:
            ex_feats_shape = initial_context["ex_feats"].shape
            if len(ex_feats_shape) not in [2, 3]:
                raise ValueError(
                    f"ex_feats must be (C, D) or (B, C, D), got shape {ex_feats_shape}"
                )

        # Initialize storage
        generated_surfaces = []
        generated_ex_feats = [] if has_ex_feats else None

        # Clone initial context to avoid modifying input
        context = {k: v.clone() if torch.is_tensor(v) else v
                   for k, v in initial_context.items()}

        for step in range(horizon):
            # Generate one-step prediction using existing method
            result = self.get_surface_given_conditions(context, z=None)

            # Handle return format (tuple for ex_feats, single tensor otherwise)
            if has_ex_feats:
                pred_surface, pred_ex_feat = result  # (B, 1, 5, 5), (B, 1, 3)
                generated_surfaces.append(pred_surface)
                generated_ex_feats.append(pred_ex_feat)

                # Extract prediction for next context
                new_surface = pred_surface[:, 0, :, :]  # (B, 5, 5)
                new_ex_feat = pred_ex_feat[:, 0, :]     # (B, 3)
            else:
                pred_surface = result  # (B, 1, 5, 5)
                generated_surfaces.append(pred_surface)

                # Extract prediction for next context
                new_surface = pred_surface[:, 0, :, :]  # (B, 5, 5)
                new_ex_feat = None

            # Update context: drop oldest, append new
            context = self._update_context(context, new_surface, new_ex_feat)

        # Concatenate all generated timesteps
        surfaces_out = torch.cat(generated_surfaces, dim=1)  # (B, horizon, 3, 5, 5)

        if has_ex_feats:
            ex_feats_out = torch.cat(generated_ex_feats, dim=1)  # (B, horizon, 3)
            return surfaces_out, ex_feats_out
        else:
            return surfaces_out

    def generate_autoregressive_offset(
        self,
        initial_context: Dict[str, torch.Tensor],
        ar_steps: int = 3,
        horizon: int = 60,
        offset: int = 60,
        z: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate multi-step autoregressive sequence using offset-based chunking.

        This method matches the training approach for phases 3-4:
        - Phase 3: ar_steps=3, horizon=60, offset=60 → 180 days total
        - Phase 4: ar_steps=3, horizon=90, offset=90 → 270 days total

        Each step generates 'horizon' days, then slides context window by 'offset' days.
        This is MUCH faster than day-by-day AR (3 calls vs 180/270 calls).

        Args:
            initial_context: dict with keys:
                - "surface": (B, C, 5, 5) - C days of context
                - "ex_feats": optional (B, C, 3) - extra features
            ar_steps: Number of AR steps (default 3)
            horizon: Days to generate per step (60 or 90)
            offset: Context window shift amount (60 or 90 for deployment)
            z: Optional pre-sampled latents (not used, model samples internally)

        Returns:
            If ex_feats in context:
                tuple of (surfaces, ex_feats)
                - surfaces: (B, ar_steps*horizon, 3, 5, 5)
                - ex_feats: (B, ar_steps*horizon, 3)
            Else:
                surfaces: (B, ar_steps*horizon, 3, 5, 5)

        Example:
            # 180-day generation
            surf, ex = model.generate_autoregressive_offset(
                context, ar_steps=3, horizon=60, offset=60
            )
            # surf.shape = (1, 180, 3, 5, 5)
        """
        # Validate inputs
        if "surface" not in initial_context:
            raise ValueError("initial_context must contain 'surface' key")

        C = self.config.get("context_len", 60)
        has_ex_feats = "ex_feats" in initial_context

        # Validate context length
        context_surface_len = initial_context["surface"].shape[1]
        if context_surface_len != C:
            raise ValueError(f"Context length must be {C}, got {context_surface_len}")

        # Clone initial context to avoid modifying input
        context = {k: v.clone() if torch.is_tensor(v) else v
                   for k, v in initial_context.items()}

        # Storage for all predictions
        all_surfaces = []
        all_ex_feats = [] if has_ex_feats else None

        # Store median predictions for context updates
        median_surfaces = []
        median_ex_feats = [] if has_ex_feats else None

        # Autoregressive rollout
        for step in range(ar_steps):
            # Temporarily set model horizon
            original_horizon = self.horizon
            self.horizon = horizon

            # Generate prediction for this step
            result = self.get_surface_given_conditions(context, z=None)

            self.horizon = original_horizon

            # Handle return format
            if has_ex_feats:
                pred_surface, pred_ex_feat = result  # (B, horizon, 5, 5), (B, horizon, 3)
                all_surfaces.append(pred_surface)
                all_ex_feats.append(pred_ex_feat)

                # Store predictions for context update
                median_surfaces.append(pred_surface)
                median_ex_feats.append(pred_ex_feat)
            else:
                pred_surface = result  # (B, horizon, 5, 5)
                all_surfaces.append(pred_surface)

                # Store predictions for context update
                median_surfaces.append(pred_surface)

            # Update context for next step (if not last step)
            if step < ar_steps - 1:
                # Concatenate all predictions so far
                concat_surfaces = torch.cat(median_surfaces, dim=1)  # (B, (step+1)*horizon, 5, 5)

                # Take last C days as new context
                context["surface"] = concat_surfaces[:, -C:, :, :]  # (B, C, 5, 5)

                if has_ex_feats:
                    concat_ex = torch.cat(median_ex_feats, dim=1)  # (B, (step+1)*horizon, 3)
                    context["ex_feats"] = concat_ex[:, -C:, :]  # (B, C, 3)

        # Concatenate all predictions
        surfaces_out = torch.cat(all_surfaces, dim=1)  # (B, ar_steps*horizon, 3, 5, 5)

        if has_ex_feats:
            ex_feats_out = torch.cat(all_ex_feats, dim=1)  # (B, ar_steps*horizon, 3)
            return surfaces_out, ex_feats_out
        else:
            return surfaces_out

    def _update_context(
        self,
        context: Dict[str, torch.Tensor],
        new_surface: torch.Tensor,
        new_ex_feat: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Update context window: drop oldest timestep, append new prediction.

        Implements sliding window: [t-C+1, ..., t] becomes [t-C+2, ..., t+1]
        Updates both surface and ex_feats (if present) to maintain consistency.

        Args:
            context: dict with "surface" (B, C, 5, 5) and optional "ex_feats" (B, C, 3)
            new_surface: (B, 5, 5) - surface to append
            new_ex_feat: optional (B, 3) - ex_feats to append (must be provided if context has ex_feats)

        Returns:
            Updated context dict with same keys and shapes as input
        """
        # Update surfaces: shift left (drop oldest), append new
        old_surfaces = context["surface"]  # (B, C, 5, 5)

        # Ensure new_surface is on same device as old_surfaces
        new_surface_device = new_surface.to(old_surfaces.device)

        new_surfaces = torch.cat([old_surfaces[:, 1:, :, :],
                                  new_surface_device.unsqueeze(1)], dim=1)

        updated = {"surface": new_surfaces}

        # Update ex_feats if present
        if "ex_feats" in context and new_ex_feat is not None:
            old_ex_feats = context["ex_feats"]  # (B, C, 3)

            # Ensure new_ex_feat is on same device as old_ex_feats
            new_ex_feat_device = new_ex_feat.to(old_ex_feats.device)

            new_ex_feats = torch.cat([old_ex_feats[:, 1:, :],
                                      new_ex_feat_device.unsqueeze(1)], dim=1)
            updated["ex_feats"] = new_ex_feats

        return updated

    def check_input(self, config: dict):
        for req in ["feat_dim", "latent_dim"]:
            if req not in config:
                raise ValueError(f"config doesn't contain: {req}")
        if "device" not in config:
            config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        if "kl_weight" not in config:
            config["kl_weight"] = 1.0
        
        for req in ["surface_hidden", "ex_feats_dim",
                    "mem_hidden", "mem_layers", 
                    "ctx_surface_hidden"]:
            if req not in config:
                raise ValueError(f"config doesn't contain {req}")
        if isinstance(config["surface_hidden"], int):
            config["surface_hidden"] = [config["surface_hidden"]]
        if isinstance(config["ctx_surface_hidden"], int):
            config["ctx_surface_hidden"] = [config["ctx_surface_hidden"]]
        
        for req in ["ex_feats_hidden", "ctx_ex_feats_hidden"]:
            if req not in config:
                config[req] = None
        if config["ex_feats_hidden"] is not None and isinstance(config["ex_feats_hidden"], int):
            config["ex_feats_hidden"] = [config["ex_feats_hidden"]]
        if config["ctx_ex_feats_hidden"] is not None and isinstance(config["ctx_ex_feats_hidden"], int):
            config["ctx_ex_feats_hidden"] = [config["ctx_ex_feats_hidden"]]
        
        if "mem_type" not in config:
            config["mem_type"] = "lstm"
        if "mem_dropout" not in config:
            config["mem_dropout"] = 0
        
        if "re_feat_weight" not in config:
            config["re_feat_weight"] = 1.0
        if "ex_feats_loss_type" not in config:
            config["ex_feats_loss_type"] = "l1"
        if "ex_loss_on_ret_only" not in config:
            config["ex_loss_on_ret_only"] = False

        if "interaction_layers" not in config:
            config["interaction_layers"] = None
        
        if "use_dense_surface" not in config:
            config["use_dense_surface"] = False
        
        if "compress_context" not in config:
            config["compress_context"] = False

    def forward(self, x: dict[str, torch.Tensor]):
        '''
            Input:
                x should be a dictionary with 2 keys:
                - surface: volatility surface of shape (B,T,H,W),
                    surface[:,:context_len,:,:] are the context surfaces (previous days),
                    surface[:,context_len:,:,:] are the surfaces to predict (H timesteps)
                - ex_feats: extra features of shape (B,T,n), not necessarily needed
            Returns:
                a tuple of reconstruction, z_mean, z_log_var, z,
                where z is sampled from distribution defined by z_mean and z_log_var
                reconstruction shape: (B, H, H, W) where H=self.horizon
        '''
        surface = x["surface"].to(self.device)
        if len(surface.shape) == 3:
            # unbatched data
            surface = surface.unsqueeze(0)
        B = surface.shape[0]
        T = surface.shape[1]
        C = T - self.horizon  # Context length = total length - horizon

        if C < 1:
            raise ValueError(f"Insufficient sequence length: T={T}, horizon={self.horizon}, "
                           f"resulting in C={C}. Need at least C >= 1.")

        ctx_surface = surface[:, :C, :, :] # c
        ctx_encoder_input = {"surface": ctx_surface}

        encoder_input = {"surface": surface}
        if "ex_feats" in x:
            ex_feats = x["ex_feats"].to(self.device)
            if len(ex_feats.shape) == 2:
                ex_feats = ex_feats.unsqueeze(0)
            ctx_ex_feats = ex_feats[:, :C, :]
            ctx_encoder_input["ex_feats"] = ctx_ex_feats
            encoder_input["ex_feats"] = ex_feats

        ctx_embedding = self.ctx_encoder(ctx_encoder_input) # embedded c (B, C, n)
        ctx_embedding_dim = ctx_embedding.shape[2]
        ctx_embedding_padded = torch.zeros((B, T, ctx_embedding_dim)).to(self.device)
        ctx_embedding_padded[:, :C, :] = ctx_embedding
        
        z_mean, z_log_var, z = self.encoder(encoder_input) # P(z|c,x), (B, T, latent_dim)

        decoder_input = torch.cat([ctx_embedding_padded, z], dim=-1)
        if "ex_feats" in x:
            decoded_surface, decoded_ex_feat = self.decoder(decoder_input) # P(x|c,z,t)
            # decoded_surface: (B, T, H, W)
            return decoded_surface[:, C:, :, :], decoded_ex_feat[:, C:, :], z_mean, z_log_var, z
        else:
            decoded_surface = self.decoder(decoder_input) # P(x|c,z,t)
            # decoded_surface: (B, T, H, W)
            return decoded_surface[:, C:, :, :], z_mean, z_log_var, z

    def train_step(self, x, optimizer: torch.optim.Optimizer):
        '''
            Input:
                x should be a dictionary with 2 keys:
                - surface: volatility surface of shape (B,T,H,W),
                    surface[:,:context_len,:,:] are the context surfaces (previous days),
                    surface[:,context_len:,:,:] are the surfaces to predict (horizon timesteps)
                - ex_feats: extra features of shape (B,T,n)
        '''

        surface = x["surface"]
        if len(surface.shape) == 3:
            # unbatched data
            surface = surface.unsqueeze(0)
        B = surface.shape[0]
        T = surface.shape[1]
        C = T - self.horizon  # Context length = total length - horizon
        surface_real = surface[:,C:, :, :].to(self.device)

        if "ex_feats" in x:
            ex_feats = x["ex_feats"]
            if len(ex_feats.shape) == 2:
                ex_feats = ex_feats.unsqueeze(0)
            ex_feats_real = ex_feats[:, C:, :,].to(self.device)

        optimizer.zero_grad()

        # Mixed precision training with BF16
        with autocast('cuda', dtype=torch.bfloat16):
            if "ex_feats" in x:
                surface_reconstruction, ex_feats_reconstruction, z_mean, z_log_var, z = self.forward(x)
            else:
                surface_reconstruction, z_mean, z_log_var, z = self.forward(x)

            # Reconstruction loss using MSE
            # surface_reconstruction: (B, horizon, H, W)
            # surface_real: (B, horizon, H, W)
            re_surface = nn.functional.mse_loss(surface_reconstruction, surface_real)
            if "ex_feats" in x:
                if self.config["ex_loss_on_ret_only"]:
                    ex_feats_reconstruction = ex_feats_reconstruction[:, :, :1]
                    ex_feats_real = ex_feats_real[:, :, :1]
                re_ex_feats = self.ex_feats_loss_fn(ex_feats_reconstruction, ex_feats_real)
                reconstruction_error = re_surface + self.config["re_feat_weight"] * re_ex_feats
            else:
                reconstruction_error = re_surface
            # KL = -1/2 \sum_{i=1}^M (1+log(\sigma_k^2) - \sigma_k^2 - \mu_k^2)
            kl_loss = -0.5 * (1 + z_log_var - torch.exp(z_log_var) - torch.square(z_mean))
            kl_loss = torch.mean(torch.sum(kl_loss, dim=-1))
            total_loss = reconstruction_error + self.kl_weight * kl_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()

        return {
            "loss": total_loss,
            "re_surface": re_surface,
            "re_ex_feats": re_ex_feats if "ex_feats" in x else torch.zeros(1),
            "reconstruction_loss": reconstruction_error,
            "kl_loss": kl_loss,
        }

    def train_step_multihorizon(self, x, optimizer: torch.optim.Optimizer,
                                horizons=[1, 7, 14, 30]):
        '''
        Train on multiple horizons simultaneously with weighted loss.

        This method trains the model to predict multiple future horizons
        [1, 7, 14, 30] days ahead in a single training step, using weighted
        loss to balance short-term and long-term prediction quality.

        Input:
            x: Dictionary with keys:
                - surface: (B, T, 5, 5) where T >= context_len + max(horizons)
                - ex_feats: (B, T, n) optional extra features
            optimizer: PyTorch optimizer
            horizons: List of horizons to train on (default: [1, 7, 14, 30])

        Returns:
            Dictionary with loss components:
                - loss: Weighted total loss across all horizons
                - reconstruction_loss: Weighted reconstruction loss
                - kl_loss: Weighted KL divergence loss
                - horizon_losses: Dict mapping each horizon to its loss

        Method:
            For each horizon h in [1, 7, 14, 30]:
                1. Temporarily set self.horizon = h
                2. Forward pass (predicts h days ahead)
                3. Compute loss with weight w[h]
                4. Accumulate weighted loss
            5. Single backward pass on accumulated loss
            6. Optimizer step

        Horizon weights (from BACKFILL_MVP_PLAN.md):
            - horizon=1:  weight=1.0 (highest priority, most accurate)
            - horizon=7:  weight=0.8
            - horizon=14: weight=0.6
            - horizon=30: weight=0.4 (lower priority, harder to predict)
        '''
        # Horizon weights - prioritize short-term predictions
        weights = {1: 1.0, 7: 0.8, 14: 0.6, 30: 0.4}

        # Validate input sequence length
        surface = x["surface"]
        if len(surface.shape) == 3:
            surface = surface.unsqueeze(0)
        T = surface.shape[1]
        max_horizon = max(horizons)
        min_context = 1  # Minimum context length

        if T < min_context + max_horizon:
            raise ValueError(f"Insufficient sequence length: T={T}, max_horizon={max_horizon}, "
                           f"need at least T >= {min_context + max_horizon}")

        # Store original horizon to restore later
        original_horizon = self.horizon

        # Accumulate losses across horizons
        total_loss = 0
        total_re = 0
        total_kl = 0
        horizon_losses = {}

        # Zero gradients once at start
        optimizer.zero_grad()

        for h in horizons:
            # Temporarily set horizon for this forward pass
            self.horizon = h

            # Forward pass (uses self.horizon internally to determine context length)
            if "ex_feats" in x:
                surface_reconstruction, ex_feats_reconstruction, z_mean, z_log_var, z = self.forward(x)
            else:
                surface_reconstruction, z_mean, z_log_var, z = self.forward(x)

            # Compute ground truth for this horizon
            C = T - h  # Context length for this horizon
            surface_real = surface[:, C:, :, :].to(self.device)

            # Reconstruction loss (MSE)
            re_surface = nn.functional.mse_loss(surface_reconstruction, surface_real)

            if "ex_feats" in x:
                ex_feats = x["ex_feats"]
                if len(ex_feats.shape) == 2:
                    ex_feats = ex_feats.unsqueeze(0)
                ex_feats_real = ex_feats[:, C:, :].to(self.device)

                if self.config["ex_loss_on_ret_only"]:
                    ex_feats_reconstruction = ex_feats_reconstruction[:, :, :1]
                    ex_feats_real = ex_feats_real[:, :, :1]

                re_ex_feats = self.ex_feats_loss_fn(ex_feats_reconstruction, ex_feats_real)
                reconstruction_error = re_surface + self.config["re_feat_weight"] * re_ex_feats
            else:
                reconstruction_error = re_surface

            # KL divergence loss
            kl_loss = -0.5 * (1 + z_log_var - torch.exp(z_log_var) - torch.square(z_mean))
            kl_loss = torch.mean(torch.sum(kl_loss, dim=-1))

            # Weighted loss for this horizon
            horizon_loss = reconstruction_error + self.kl_weight * kl_loss
            weight = weights.get(h, 1.0)  # Default to 1.0 if horizon not in weights dict
            weighted_loss = weight * horizon_loss

            # Accumulate
            total_loss += weighted_loss
            total_re += weight * reconstruction_error.item()
            total_kl += weight * kl_loss.item()
            horizon_losses[h] = horizon_loss.item()

        # Restore original horizon
        self.horizon = original_horizon

        # Single backward pass with accumulated gradients
        total_loss.backward()
        optimizer.step()

        return {
            "loss": total_loss.item(),
            "reconstruction_loss": total_re,
            "kl_loss": total_kl,
            "horizon_losses": horizon_losses,  # Per-horizon loss for logging
        }

    def test_step(self, x):
        surface = x["surface"]
        if len(surface.shape) == 3:
            # unbatched data
            surface = surface.unsqueeze(0)
        B = surface.shape[0]
        T = surface.shape[1]
        C = T - self.horizon  # Context length = total length - horizon
        surface_real = surface[:, C:, :, :].to(self.device)

        if "ex_feats" in x:
            ex_feats = x["ex_feats"]
            if len(ex_feats.shape) == 2:
                ex_feats = ex_feats.unsqueeze(0)
            ex_feats_real = ex_feats[:, C:, :,].to(self.device)

        if "ex_feats" in x:
            surface_reconstruction, ex_feats_reconstruction, z_mean, z_log_var, z = self.forward(x)
        else:
            surface_reconstruction, z_mean, z_log_var, z = self.forward(x)

        # Reconstruction loss using MSE
        # surface_reconstruction: (B, horizon, H, W)
        # surface_real: (B, horizon, H, W)
        re_surface = nn.functional.mse_loss(surface_reconstruction, surface_real)
        if "ex_feats" in x:
            if self.config["ex_loss_on_ret_only"]:
                ex_feats_reconstruction = ex_feats_reconstruction[:, :, :1]
                ex_feats_real = ex_feats_real[:, :, :1]
            re_ex_feats = self.ex_feats_loss_fn(ex_feats_reconstruction, ex_feats_real)
            reconstruction_error = re_surface + self.config["re_feat_weight"] * re_ex_feats
        else:
            reconstruction_error = re_surface
        # KL = -1/2 \sum_{i=1}^M (1+log(\sigma_k^2) - \sigma_k^2 - \mu_k^2)
        kl_loss = -0.5 * (1 + z_log_var - torch.exp(z_log_var) - torch.square(z_mean))
        kl_loss = torch.mean(torch.sum(kl_loss, dim=-1))
        total_loss = reconstruction_error + self.kl_weight * kl_loss

        return {
            "loss": total_loss,
            "re_surface": re_surface,
            "re_ex_feats": re_ex_feats if "ex_feats" in x else torch.zeros(1),
            "reconstruction_loss": reconstruction_error,
            "kl_loss": kl_loss,
        }