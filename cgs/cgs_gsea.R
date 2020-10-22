#!/usr/bin/env Rscript
# XPR GSEA

args = commandArgs(trailingOnly=TRUE)
suffix = args[1]
cgs_data_dir = args[2]
cgs_info_dir = args[3]

print(paste('Split:',suffix))
print(paste('data_dir:',cgs_data_dir))
print(paste('info_dir:',cgs_info_dir))

# get XPR data
# this version includes num_label


sh_cgs_data = read.table(file = cgs_data_dir, sep = '\t', header = T)
sh_cgs_info = read.table(file = cgs_info_dir, sep = '\t', header = T)

rownames(sh_cgs_data)=sh_cgs_info$sig_id
xpr_data = read.table(file = 'xpr_data_tst.txt',sep = '\t', header = T)
xpr_data = as.matrix(xpr_data[,1:978])
xpr_info = read.table(file = 'xpr_inst_info_tst.txt',sep = '\t', header = T)

# GSEA

up_ls = apply(xpr_data,1,function(x)colnames(xpr_data)[order(x,decreasing=T)[1:50]])
up_ls = data.frame(up_ls)
names(up_ls)=xpr_info$inst_id

dn_ls = apply(xpr_data,1,function(x)colnames(xpr_data)[order(x,decreasing=F)[1:50]])
dn_ls = data.frame(dn_ls)
names(dn_ls)=xpr_info$inst_id

library(fgsea)

get_gsea=function(data,pathways){
  pval=matrix(NA, nrow = dim(data)[1], ncol = length(pathways))
  colnames(pval)=names(pathways)
  rownames(pval)=rownames(data)
  padj=matrix(NA, nrow = dim(data)[1], ncol = length(pathways))
  colnames(padj)=names(pathways)
  rownames(padj)=rownames(data)
  es=matrix(NA, nrow = dim(data)[1], ncol = length(pathways))
  colnames(es)=names(pathways)
  rownames(es)=rownames(data)
  data=as.matrix(data)
  for (i in 1:dim(data)[1]){
    q=sort(data[i,],decreasing=T)
    q_gsea=fgsea(stats=q,pathways=pathways,nperm=1000)
    rownames(q_gsea)=q_gsea$pathway
    stopifnot(colnames(pval)==q_gsea$pathway,colnames(padj)==q_gsea$pathway,colnames(es)==q_gsea$pathway)
    pval[i,]=q_gsea$pval
    padj[i,]=q_gsea$padj
    es[i,]=q_gsea$ES
    if (i%%100==0){
      print(paste('Progress:','query',i))
    }
  }
  res=list(pval,padj,es)
  names(res)=c('pval','padj','es')
  return(res)
}

gsea_res=get_gsea(sh_cgs_data,up_ls)
write.table(gsea_res$pval,file =paste(suffix,'xpr_up_pval.txt',sep='_'),
            sep='\t',col.names = NA)
write.table(gsea_res$padj,file =paste(suffix,'xpr_up_padj.txt',sep='_'),
            sep='\t',col.names = NA)
write.table(gsea_res$es,file =paste(suffix,'xpr_up_es.txt',sep='_')
            ,sep='\t',col.names = NA)

print('ESup finished!')

gsea_res=get_gsea(sh_cgs_data,dn_ls)
write.table(gsea_res$pval,file =paste(suffix,'xpr_dn_pval.txt',sep='_'),
            sep='\t',col.names = NA)
write.table(gsea_res$padj,file =paste(suffix,'xpr_dn_padj.txt',sep='_'),
            sep='\t',col.names = NA)
write.table(gsea_res$es,file =paste(suffix,'xpr_dn_es.txt',sep='_')
            ,sep='\t',col.names = NA)

print('ESdown finished!')

