import numpy as np

class DOATrackerSingle:
    def __init__(self, config_dict):
        """
        Tracker DOA de Fuente Única basado en un sistema de 'Salud' (Batería).
        Se inicializa usando el diccionario TRACKER definido en config.py.
        """
        # Parámetros extraídos del diccionario de configuración
        self.alpha = config_dict['alpha']
        self.gate_deg = config_dict['gate_deg']
        self.conf_keep = config_dict['conf_keep']
        self.conf_start = config_dict['conf_start']
        self.health_max = config_dict['health_max']
        self.health_damage = config_dict['health_damage']
        self.min_age_confirm = config_dict['min_age_confirm']
        
        # Estado (Memoria interna del tracker)
        self.tracks = []
        self.next_id = 1

    def actualizar(self, measurements, confidences):
        """
        Recibe las estimaciones instantáneas (ángulos y fiabilidad) 
        del estimador MUSIC y actualiza la memoria de las pistas.
        
        Retorna:
        - out_angles: Lista vacía [] si no hay fuente fiable confirmada, 
                      o una lista con 1 solo ángulo [angulo_oficial] dominante.
        """
        # Copiamos las listas para poder extraer elementos (pop) a medida que los asignamos
        unassigned_meas = list(measurements)
        unassigned_conf = list(confidences)

        # =========================================================
        # A. MANTENIMIENTO: Asociar medidas actuales a pistas existentes
        # =========================================================
        for track in self.tracks:
            # Filtramos los índices de las medidas que superan el umbral para mantener una pista
            valid_indices = [i for i, conf in enumerate(unassigned_conf) if conf >= self.conf_keep]
            
            if not valid_indices:
                # Si ninguna medida nos vale, la pista actual pierde salud (batería)
                track['health'] -= self.health_damage
                continue
                
            # Buscar la medida angularmente más cercana (distancia circular mínima)
            min_dist = float('inf')
            best_sub_idx = -1
            
            for idx in valid_indices:
                meas = unassigned_meas[idx]
                # Calculamos la distancia por el camino más corto en una circunferencia de 360º
                dist = min(abs(meas - track['angle']), 360 - abs(meas - track['angle']))
                if dist < min_dist:
                    min_dist = dist
                    best_sub_idx = idx
                    
            # Si la medida más cercana está dentro del radio de captura (gate)
            if min_dist <= self.gate_deg:
                best_idx = best_sub_idx
                
                # 1. Filtro Alpha Dinámico (más suave cuanta menos fiabilidad tenga la medida)
                alpha_dinamico = self.alpha * unassigned_conf[best_idx]
                
                # 2. Calculamos la diferencia angular por el camino más corto
                diff = unassigned_meas[best_idx] - track['angle']
                if diff > 180:
                    diff -= 360
                elif diff < -180:
                    diff += 360
                    
                # 3. Actualizamos la posición
                track['angle'] = (track['angle'] + alpha_dinamico * diff) % 360
                
                # 4. RECOMPENSA: Curamos la pista acumulando la fiabilidad (hasta el tope máximo)
                track['health'] = min(self.health_max, track['health'] + unassigned_conf[best_idx])
                track['age'] += 1
                
                # 5. Consumimos la medida para que otra pista no pueda usarla
                unassigned_meas.pop(best_idx)
                unassigned_conf.pop(best_idx)
            else:
                # Si la medida más cercana está demasiado lejos, la pista pierde salud
                track['health'] -= self.health_damage

        # =========================================================
        # B. NACIMIENTO: Crear nuevas pistas con las medidas sobrantes
        # =========================================================
        # Solo nacen pistas de medidas que superan el umbral estricto (conf_start)
        valid_start_indices = [i for i, conf in enumerate(unassigned_conf) if conf >= self.conf_start]
        
        for idx in valid_start_indices:
            new_track = {
                'id': self.next_id,
                'angle': unassigned_meas[idx],
                'age': 1,
                'health': unassigned_conf[idx]  # Nace con una salud inicial igual a su fiabilidad
            }
            self.tracks.append(new_track)
            self.next_id += 1

        # =========================================================
        # C. SALIDA OFICIAL: "El Rey de la Colina" (Single Source)
        # =========================================================
        out_angles = []
        best_track_idx = -1
        max_health = float('-inf')
        
        for i, track in enumerate(self.tracks):
            # Para competir por salir en pantalla, la pista debe ser madura y estar viva
            if track['age'] >= self.min_age_confirm and track['health'] > 0:
                
                # Batalla de salud: Gana la pista que tenga la barra de vida más llena
                if track['health'] > max_health:
                    max_health = track['health']
                    best_track_idx = i
                    
        # Añadimos solo al ganador indiscutible a la salida
        if best_track_idx != -1:
            out_angles.append(self.tracks[best_track_idx]['angle'])

        # =========================================================
        # D. LIMPIEZA: Eliminar pistas muertas de la memoria
        # =========================================================
        # Destruimos las pistas cuya salud haya caído a 0 o por debajo
        self.tracks = [t for t in self.tracks if t['health'] > 0]
        
        return out_angles

    def reset(self):
        """
        Función de utilidad para limpiar la memoria del tracker 
        sin necesidad de volver a instanciar la clase.
        """
        self.tracks = []
        self.next_id = 1