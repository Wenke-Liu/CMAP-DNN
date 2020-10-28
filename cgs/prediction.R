#!/usr/bin/env Rscript

message('Reading ES data...')
es=read.table(file='cgs_xpr_wtcs.txt',header=T,row.names=1,sep='\t')
message('Done!')

sig_id=read.table(file='sig_id.txt',header=F,sep='\t')
#sh_cgs_info=read.table(file='sh_cgs_info.txt',header=T,sep='\t')
colnames(sig_id)='sig_id'
sig_id=cbind(sig_id,gene=gsub('.*_','',sig_id$sig_id))
es=es[as.character(sig_id$sig_id),]
es_ave = apply(es,2,function(x)tapply(x,sig_id$gene,mean))

xpr_info = read.table(file = 'xpr_inst_info_tst.txt',sep = '\t', header = T)

es_max = apply(es,2,function(x)tapply(x,sig_id$gene,max))

## check prediction
esm_gsea_pred = apply(es_max,2,function(x)rownames(es_max)[which.max(x)])
print('Number of correct predictions:')
sum(esm_gsea_pred==xpr_info$pert_iname)

## softmax
softmax=function(x){
  y = max(x)
  logsumexp = y + log(sum(exp(x - y)))
  return(exp(x - logsumexp))
}

esm_gsea_score = apply(esm_gsea_pred, 2, softmax)

write.table(esm_gsea_score,
            file='cgs_xpr_gsea_max_prob.txt',
            sep='\t',col.names = NA,quote=F)

scores = read.table(file='cgs_xpr_gsea_gene_max_score.txt', header=T, row.names=1, sep='\t')

num_label = read.table('sh_num_label.txt', header=F, sep='\t')
lab_dict = num_label$V2
names(lab_dict) = num_label$V1
probs = apply(scores,2,softmax)

setdiff(as.character(num_label$V1), colnames(probs))

probs = cbind(probs, GEMIN2 = 0, KIF13B = 0)
probs = probs[,as.character(num_label$V1)]

all(gsub(':', '.',as.character(xpr_info$inst_id)) == rownames(probs))
probs = cbind(probs, num_label = xpr_info$num_label)

write.table(probs,
            file='cgs_xpr_gsea_max_prob_softmax.txt',
            sep='\t', row.names=F,quote=F)