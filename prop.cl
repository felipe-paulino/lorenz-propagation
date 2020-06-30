
//-------------------------------------------
//definicao das equacoes de lorenz
//-------------------------------------------

//derivada temporal de x
float dxdt(x,y,z,sigma,rho,beta){
	return sigma*(y-x);	
}

//derivada temporal de y
float dydt(x,y,z,sigma,rho,beta){
	return x*(rho-z)-y;
}

//derivada temporal de x
float dzdt(x,y,z,sigma,rho,beta){
	return x*y-beta*z;	
}
//-------------------------------------------

__kernel void prop(
	const int numSteps,
    const int N,
	const float sigma,
	const float rho,
	const float beta,
	const float step,
    __global float* x,
    __global float* y,
    __global float* z)
{
	float k1x,k1y,k1z;
	float k2x,k2y,k2z;
	float k3x,k3y,k3z;
	float k4x,k4y,k4z;
	
	int i = get_global_id(0);
	int k;
	
	//na primeira iteracao o ultimo bloco de n_estados (estados do ultimo instante calculado) eh usado para calcular
    //o primeiro bloco de n_estados (primeiro instante da fila). 
	//Depois o primeiro instante é usado para calcular o segundo e assim por diante
	for(k=0;k<numSteps;k++){
		int oldIdx,newIdx;
		if(k==0){
			oldIdx=((numSteps-1)*N)+i;
			newIdx=i;
		}else{
			oldIdx=(k-1)*N+i;
			newIdx=k*N+i;
		}
		k1x = step * dxdt(x[oldIdx],y[oldIdx],z[oldIdx],sigma,rho,beta);
		k1y = step * dydt(x[oldIdx],y[oldIdx],z[oldIdx],sigma,rho,beta); 
		k1z = step * dzdt(x[oldIdx],y[oldIdx],z[oldIdx],sigma,rho,beta); 
		
		k2x = step * dxdt(x[oldIdx] + 0.5*k1x, y[oldIdx] + 0.5*k1y, z[oldIdx] + 0.5*k1z,sigma,rho,beta);
		k2y = step * dydt(x[oldIdx] + 0.5*k1x, y[oldIdx] + 0.5*k1y, z[oldIdx] + 0.5*k1z,sigma,rho,beta);
		k2z = step * dzdt(x[oldIdx] + 0.5*k1x, y[oldIdx] + 0.5*k1y, z[oldIdx] + 0.5*k1z,sigma,rho,beta);
		
		k3x = step * dxdt(x[oldIdx] + 0.5*k2x, y[oldIdx]+0.5*k2y, z[oldIdx]+0.5*k2z, sigma, rho, beta);
		k3y = step * dydt(x[oldIdx] + 0.5 * k2x,y[oldIdx] + 0.5 * k2y,z[oldIdx] + 0.5 * k2z,sigma,rho,beta);
		k3z = step * dzdt(x[oldIdx] + 0.5 * k2x,y[oldIdx] + 0.5 * k2y,z[oldIdx] + 0.5 * k2z,sigma,rho,beta);
		
		k4x = step * dxdt(x[oldIdx] + k3x,y[oldIdx]+k3y,z[oldIdx]+k3z,sigma,rho,beta);
		k4y = step * dydt(x[oldIdx] + k3x,y[oldIdx]+k3y,z[oldIdx]+k3z,sigma,rho,beta);
		k4z = step * dzdt(x[oldIdx] + k3x,y[oldIdx]+k3y,z[oldIdx]+k3z,sigma,rho,beta);
		
		x[newIdx]=x[oldIdx]+(1.0/6.0)*(k1x + 2*k2x + 2*k3x + k4x);
		y[newIdx]=y[oldIdx]+(1.0/6.0)*(k1y + 2*k2y + 2*k3y + k4y);
		z[newIdx]=z[oldIdx]+(1.0/6.0)*(k1z + 2*k2z + 2*k3z + k4z);
	}
	
}
