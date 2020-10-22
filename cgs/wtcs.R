#!/usr/bin/env Rscript

up_es = read.table(file='cgs_xpr_gsea_up_es.txt',header=T, row.names=1,sep='\t')
message('ESup loaded!')
up_es = as.matrix(up_es)
up_es_sgn = sign(up_es)
dn_es = read.table(file='cgs_xpr_gsea_dn_es.txt',header=T, row.names=1,sep='\t')
dn_es = as.matrix(dn_es)
message('ESdown loaded!')
dn_es_sgn = sign(dn_es)
sgn = (up_es_sgn-dn_es_sgn)!=0
es = (up_es-dn_es)/2*sgn
message('writing wtcs file...')
write.table(es,file='cgs_xpr_wtcs.txt',
            sep='\t',col.names = NA,quote=F)
message('done!')